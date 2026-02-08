import streamlit as st
import pandas as pd
import json
import os
import calendar
from datetime import datetime
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go

import zipfile
import tempfile
import shutil

# Set page config
st.set_page_config(
    page_title="Discord Data Viewer",
    page_icon="💬",
    layout="wide"
)

# Initialize session state for data path
if 'data_path' not in st.session_state:
    st.session_state.data_path = None
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = None

# Cleanup function for temp dir
def cleanup_temp_dir():
    if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
        try:
            shutil.rmtree(st.session_state.temp_dir)
            st.session_state.temp_dir = None
            st.session_state.data_path = None
        except Exception as e:
            st.error(f"Error cleaning up temporary files: {e}")

# Register cleanup on script exit (best effort)
import atexit
atexit.register(cleanup_temp_dir)


# Initialize session state
if 'selected_channel' not in st.session_state:
    st.session_state.selected_channel = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Messages"

@st.cache_data
def load_channels(base_path):
    """Loads the channel index from index.json."""
    index_file = os.path.join(base_path, "messages", "index.json")
    
    # Try different casing if uppercase fails, or check directly in root
    if not os.path.exists(index_file):
         # Try "Mensajes" (Spanish export)
        index_file = os.path.join(base_path, "Mensajes", "index.json")
        
    if not os.path.exists(index_file):
        # Try root
        index_file = os.path.join(base_path, "index.json")

    if not os.path.exists(index_file):
        # Scan for it
        for root, dirs, files in os.walk(base_path):
            if "index.json" in files:
                index_file = os.path.join(root, "index.json")
                break
    
    if not os.path.exists(index_file):
        return {}, None
    
    # Return both data and the actual messages directory path
    messages_dir = os.path.dirname(index_file)
    
    with open(index_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            return data, messages_dir
        except json.JSONDecodeError:
            st.error("Error decoding index.json")
            return {}, None

@st.cache_data
def get_channel_stats(channels, base_path):
    """
    Iterates through all channels to count messages.
    Returns a DataFrame with Channel ID, Name, Count, Type (DM/Server).
    """
    stats = []
    
    # Create a progress bar because this might take a moment
    progress_bar = st.progress(0)
    total_channels = len(channels)
    
    for i, (channel_id, channel_name) in enumerate(channels.items()):
        # Update progress every 10 channels to avoid too many UI updates
        if i % 10 == 0:
            progress_bar.progress((i + 1) / total_channels)
            
        # The folder names have a 'c' prefix
        messages_file = os.path.join(base_path, f"c{channel_id}", "messages.json")
        count = 0
        if os.path.exists(messages_file):
            try:
                # We just need the count, so we can load it. 
                # If files are huge, we might want to stream it, but JSON usually requires full load.
                with open(messages_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        count = len(data)
            except:
                pass # Ignore errors for stats
        
        # Determine type based on name heuristic
        # DMs usually start with "Direct Message with"
        # Server channels usually have " in " (e.g. "general in ServerName")
        if "Direct Message with" in channel_name:
            c_type = "Direct Message"
            # Clean name for DMs
            display_name = channel_name.replace("Direct Message with ", "").strip()
        elif " in " in channel_name:
            c_type = "Server"
            display_name = channel_name
        else:
            c_type = "Unknown"
            display_name = channel_name

        stats.append({
            "Channel ID": channel_id,
            "Full Name": channel_name,
            "Display Name": display_name,
            "Message Count": count,
            "Type": c_type
        })
    
    progress_bar.empty()
    
    df = pd.DataFrame(stats)
    return df

@st.cache_data
def load_messages(channel_id, base_path):
    """Loads messages for a specific channel ID."""
    # The folder names have a 'c' prefix
    messages_file = os.path.join(base_path, f"c{channel_id}", "messages.json")
    if not os.path.exists(messages_file):
        return []
    
    with open(messages_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return []
        except json.JSONDecodeError:
            return []

def get_user_stats(messages):
    """Extract user statistics from messages."""
    if not messages:
        return pd.DataFrame()
    
    df_msgs = pd.DataFrame(messages)
    
    # Ensure Timestamp is datetime
    if 'Timestamp' in df_msgs.columns:
        df_msgs['Timestamp'] = pd.to_datetime(df_msgs['Timestamp'])
    
    # Count messages per author
    if 'ID' in df_msgs.columns:
        author_stats = df_msgs.groupby('ID').agg({
            'Contents': 'count',
            'Timestamp': ['min', 'max']
        }).reset_index()
        
        author_stats.columns = ['Author', 'Message Count', 'First Message', 'Last Message']
        author_stats['Percentage'] = (author_stats['Message Count'] / len(df_msgs) * 100).round(2)
        author_stats = author_stats.sort_values('Message Count', ascending=False)
        
        return author_stats
    
    return pd.DataFrame()

@st.cache_data
def get_activity_by_period(channels, base_path, grouping='day'):
    """
    Iterates through all channels to extract message timestamps.
    Returns a DataFrame suitable for plotting: Date, Channel, Count.
    
    Args:
        channels: Dictionary of channel IDs and names
        grouping: 'day', 'month', or 'year' for time grouping
    """
    all_data = []
    
    # Progress UI
    progress_text = "Analyzing timeline data..."
    my_bar = st.progress(0, text=progress_text)
    total_channels = len(channels)
    
    for i, (channel_id, channel_name) in enumerate(channels.items()):
        # Update progress occasionally
        if i % 10 == 0:
            my_bar.progress((i + 1) / total_channels, text=f"{progress_text} ({i}/{total_channels})")
            
        messages_file = os.path.join(base_path, f"c{channel_id}", "messages.json")
        
        if os.path.exists(messages_file):
            try:
                with open(messages_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list) and data:
                        # Extract timestamps
                        df_temp = pd.DataFrame(data)
                        if 'Timestamp' in df_temp.columns:
                            # Convert to datetime
                            df_temp['Timestamp'] = pd.to_datetime(df_temp['Timestamp'])
                            
                            # Group by selected period
                            if grouping == 'day':
                                grouped_counts = df_temp.groupby(df_temp['Timestamp'].dt.date).size()
                                date_converter = lambda d: pd.Timestamp(d)
                            elif grouping == 'month':
                                grouped_counts = df_temp.groupby(df_temp['Timestamp'].dt.to_period('M')).size()
                                date_converter = lambda p: p.to_timestamp()
                            elif grouping == 'year':
                                grouped_counts = df_temp.groupby(df_temp['Timestamp'].dt.year).size()
                                date_converter = lambda y: pd.Timestamp(year=y, month=1, day=1)
                            else:
                                grouped_counts = df_temp.groupby(df_temp['Timestamp'].dt.date).size()
                                date_converter = lambda d: pd.Timestamp(d)
                            
                            for period, count in grouped_counts.items():
                                channel_display_name = channel_name
                                if "Direct Message with " in channel_name:
                                    channel_display_name = channel_name.replace("Direct Message with ", "")
                                
                                all_data.append({
                                    "Date": date_converter(period),
                                    "Channel": channel_display_name,
                                    "Count": count
                                })
            except Exception:
                pass # Skip problematic files
                
    my_bar.empty()
    
    if not all_data:
        return pd.DataFrame(columns=["Date", "Channel", "Count"])
        
    return pd.DataFrame(all_data)

def render_timeline_chart(timeline_df, grouping='day'):
    """Render the Plotly interactive timeline chart."""
    if timeline_df.empty:
        st.info("No timeline data available.")
        return

    st.subheader("📈 Message History Timeline")
    
    # Grouping labels
    grouping_labels = {
        'day': 'Date',
        'month': 'Month',
        'year': 'Year'
    }
    date_label = grouping_labels.get(grouping, 'Date')
    
    # Create line chart
    fig = px.line(
        timeline_df, 
        x="Date", 
        y="Count", 
        color="Channel",
        markers=True,
        title=f"Messages over Time by Channel ({grouping.capitalize()})",
        labels={"Count": "Number of Messages", "Date": date_label},
        template="plotly_white"
    )
    
    # Update layout for better UX with dynamic range selectors based on grouping
    if grouping == 'day':
        range_buttons = [
            dict(count=7, label="1w", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ]
    elif grouping == 'month':
        range_buttons = [
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=3, label="3y", step="year", stepmode="backward"),
            dict(step="all")
        ]
    else:  # year
        range_buttons = [
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(count=10, label="10y", step="year", stepmode="backward"),
            dict(step="all")
        ]
    
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(buttons=range_buttons),
            rangeslider=dict(visible=True),
            type="date"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_dashboard(df_stats, channels_index, base_path):
    """Render the main dashboard view."""
    st.title("📂 Discord Data Viewer")
    
    # Summary Statistics Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Channels", len(df_stats))
    with col2:
        st.metric("Total Messages", f"{df_stats['Message Count'].sum():,}")
    with col3:
        dm_count = len(df_stats[df_stats['Type'] == 'Direct Message'])
        st.metric("Direct Messages", dm_count)
    with col4:
        server_count = len(df_stats[df_stats['Type'] == 'Server'])
        st.metric("Server Channels", server_count)
    
    st.markdown("---")

    # Timeline Chart with grouping selector
    st.subheader("📊 Time Grouping")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        grouping_option = st.selectbox(
            "Group by:",
            options=['day', 'month', 'year'],
            format_func=lambda x: {'day': '📅 Day', 'month': '📆 Month', 'year': '🗓️ Year'}[x],
            key="timeline_grouping"
        )
    
    with st.spinner("Loading timeline data..."):
        timeline_df = get_activity_by_period(channels_index, base_path, grouping=grouping_option)
        render_timeline_chart(timeline_df, grouping=grouping_option)

    st.markdown("---")
    
    # Search Bar - Wide and Prominent
    st.subheader("🔍 Search Channels")
    search_col1, search_col2 = st.columns([3, 1])
    
    with search_col1:
        search_query = st.text_input(
            "Search by channel name",
            "",
            placeholder="Type to search channels...",
            label_visibility="collapsed"
        )
    
    with search_col2:
        filter_type = st.selectbox(
            "Filter",
            ["All", "Direct Messages", "Server Channels"],
            label_visibility="collapsed"
        )
    
    # Apply Filters
    filtered_df = df_stats.copy()
    
    if filter_type == "Direct Messages":
        filtered_df = filtered_df[filtered_df["Type"] == "Direct Message"]
    elif filter_type == "Server Channels":
        filtered_df = filtered_df[filtered_df["Type"] == "Server"]
    
    if search_query:
        filtered_df = filtered_df[filtered_df["Full Name"].str.contains(search_query, case=False, na=False)]
    
    filtered_df = filtered_df.sort_values("Message Count", ascending=False)
    
    st.markdown("---")
    
    # Channel Cards - Centered Window
    st.subheader("📊 Channels Overview")
    
    if len(filtered_df) == 0:
        st.info("No channels found matching your search criteria.")
    else:
        # Display top channels as cards
        st.markdown("### Top Channels")
        
        # Show top 6 channels as cards
        top_channels = filtered_df.head(6)
        
        cols = st.columns(3)
        for idx, (_, row) in enumerate(top_channels.iterrows()):
            with cols[idx % 3]:
                icon = "💬" if row['Type'] == "Direct Message" else "📢"
                
                # Create a card-like container
                with st.container():
                    st.markdown(f"""
                    <div style="
                        border: 2px solid #e0e0e0;
                        border-radius: 10px;
                        padding: 15px;
                        margin: 10px 0;
                        background-color: #f9f9f9;
                        cursor: pointer;
                        transition: all 0.3s;
                    ">
                        <h4 style="margin: 0; color: #1f77b4;">{icon} {row['Display Name'][:30]}...</h4>
                        <p style="margin: 5px 0; color: #666; font-size: 0.9em;">{row['Type']}</p>
                        <p style="margin: 5px 0; font-size: 1.2em; font-weight: bold; color: #2ca02c;">
                            {row['Message Count']:,} messages
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"View Channel", key=f"btn_{row['Channel ID']}"):
                        st.session_state.selected_channel = row['Channel ID']
                        st.rerun()
        
        st.markdown("---")
        
        # Full Statistics Table
        st.subheader("📋 All Channels Statistics")
        
        # Make the table interactive
        display_df = filtered_df[["Display Name", "Type", "Message Count"]].copy()
        display_df["Message Count"] = display_df["Message Count"].apply(lambda x: f"{x:,}")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            height=400
        )
        
        # Add selection from table
        st.markdown("**Select a channel to view details:**")
        channel_options = filtered_df["Display Name"].tolist()
        selected_name = st.selectbox(
            "Choose a channel",
            [""] + channel_options,
            format_func=lambda x: "Select a channel..." if x == "" else x
        )
        
        if selected_name:
            channel_id = filtered_df[filtered_df["Display Name"] == selected_name]["Channel ID"].iloc[0]
            st.session_state.selected_channel = channel_id
            st.rerun()

def render_channel_view(channel_id, df_stats, base_path):
    """Render the detailed channel view with tabs."""
    row = df_stats[df_stats["Channel ID"] == channel_id].iloc[0]
    
    # Back button
    if st.button("← Back to Dashboard", type="primary"):
        st.session_state.selected_channel = None
        st.rerun()
    
    st.title(f"💬 {row['Display Name']}")
    st.caption(f"Type: {row['Type']} | Total Messages: {row['Message Count']:,}")
    
    st.markdown("---")
    
    # Load Messages
    messages = load_messages(channel_id, base_path)
    
    if not messages:
        st.info("No messages found in this channel.")
        return
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📨 Messages", "📅 Calendar", "👥 People"])
    
    # Convert to DataFrame once
    df_msgs = pd.DataFrame(messages)
    if 'Timestamp' in df_msgs.columns:
        df_msgs['Timestamp'] = pd.to_datetime(df_msgs['Timestamp'])
        df_msgs = df_msgs.sort_values('Timestamp')
    
    # TAB 1: Messages
    with tab1:
        st.subheader("Messages")
        
        # Message Search
        msg_search = st.text_input("🔍 Search messages in this channel", "", key="msg_search")
        
        display_msgs = df_msgs.copy()
        if msg_search:
            display_msgs = display_msgs[display_msgs['Contents'].str.contains(msg_search, case=False, na=False)]
            st.write(f"Found **{len(display_msgs)}** matches")
        
        # Display Messages
        if len(display_msgs) == 0:
            st.info("No messages found.")
        else:
            # Pagination
            messages_per_page = 50
            total_pages = (len(display_msgs) - 1) // messages_per_page + 1
            
            page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
            
            start_idx = (page - 1) * messages_per_page
            end_idx = min(start_idx + messages_per_page, len(display_msgs))
            
            st.caption(f"Showing messages {start_idx + 1} to {end_idx} of {len(display_msgs)}")
            
            for index, msg_row in display_msgs.iloc[start_idx:end_idx].iterrows():
                with st.chat_message("user"): 
                    timestamp_str = msg_row['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    st.write(f"**{timestamp_str}**")
                    
                    if msg_row['Contents']:
                        st.markdown(msg_row['Contents'])
                    
                    if msg_row['Attachments']:
                        st.image(msg_row['Attachments'], caption="Attachment", width=300)
    
    # TAB 2: Calendar
    with tab2:
        st.subheader("📅 Message Calendar")
        
        # Prepare data for calendar
        df_msgs['Date'] = df_msgs['Timestamp'].dt.date
        active_dates = set(df_msgs['Date'])
        
        # Determine default year/month (last message)
        last_msg_date = df_msgs['Timestamp'].max()
        default_year = last_msg_date.year
        
        # Controls for Year
        years = sorted(list(set(d.year for d in active_dates)))
        if not years:
            years = [datetime.now().year]
        
        try:
            year_idx = years.index(default_year)
        except ValueError:
            year_idx = 0
            
        selected_year = st.selectbox("Year", years, index=year_idx, key="cal_year")

        # Custom Calendar Class
        class MessageCalendar(calendar.HTMLCalendar):
            def __init__(self, active_dates, year):
                super().__init__()
                self.active_dates = active_dates
                self.year = year

            def formatday(self, day, weekday):
                if day == 0:
                    return '<td class="noday">&nbsp;</td>'
                
                current_date = datetime(self.year, self.current_month, day).date()
                
                if current_date in self.active_dates:
                    return f'<td style="background-color: #90EE90; color: black; font-weight: bold; text-align: center; border: 1px solid #ddd; padding: 4px; font-size: 0.8em;">{day}</td>'
                else:
                    return f'<td style="background-color: white; color: black; text-align: center; border: 1px solid #ddd; padding: 4px; font-size: 0.8em;">{day}</td>'

            def formatweek(self, theweek):
                s = "".join(self.formatday(d, wd) for (d, wd) in theweek)
                return f"<tr>{s}</tr>"

            def formatmonth(self, year, month):
                self.current_month = month
                self.year = year
                
                s = f'<table border="0" cellpadding="0" cellspacing="0" class="month" style="width: 100%; border-collapse: collapse; margin-bottom: 10px;">\n'
                s += self.formatmonthname(year, month, withyear=False)
                s += self.formatweekheader()
                for week in self.monthdays2calendar(year, month):
                    s += self.formatweek(week)
                s += "</table>"
                return s

        # Render Calendar Grid
        cal = MessageCalendar(active_dates, selected_year)
        
        # Create 3 columns for the grid (4 rows x 3 columns = 12 months)
        cols = st.columns(3)
        
        for month in range(1, 13):
            with cols[(month - 1) % 3]:
                html_cal = cal.formatmonth(selected_year, month)
                st.markdown(html_cal, unsafe_allow_html=True)
    
    # TAB 3: People
    with tab3:
        st.subheader("👥 People in this Channel")
        
        user_stats = get_user_stats(messages)
        
        if user_stats.empty:
            st.info("No user statistics available.")
        else:
            # Display summary
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Participants", len(user_stats))
            with col2:
                top_user = user_stats.iloc[0]
                st.metric("Most Active User", f"{top_user['Author']}")
            
            st.markdown("---")
            
            # Display user table
            st.dataframe(
                user_stats,
                use_container_width=True,
                height=400
            )
            
            # Bar chart of top users
            st.subheader("Top 10 Most Active Users")
            top_10 = user_stats.head(10)
            st.bar_chart(top_10.set_index('Author')['Message Count'])

def main():
    st.sidebar.title("Data Source")
    
    # File Uploader
    uploaded_file = st.sidebar.file_uploader("Upload Discord Data Package (ZIP)", type="zip")
    
    # Process Upload
    if uploaded_file is not None:
        # Check if we need to extract
        if st.session_state.data_path is None:
            with st.spinner("Extracting ZIP file securely to temporary local storage..."):
                try:
                    # Create a temporary directory
                    temp_dir = tempfile.mkdtemp()
                    st.session_state.temp_dir = temp_dir
                    
                    # Extract zip
                    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    st.session_state.data_path = temp_dir
                    st.sidebar.success("Data loaded successfully!")
                except Exception as e:
                    st.sidebar.error(f"Error processing ZIP: {e}")
                    return

    # Option to use local folder (for debugging or pre-extracted)
    # local_path = st.sidebar.text_input("Or path to local 'messages' folder", value="")
    # if local_path and os.path.exists(local_path):
    #     st.session_state.data_path = local_path

    # Main Content
    if st.session_state.data_path:
        # Load Channels
        channels_index, messages_dir = load_channels(st.session_state.data_path)
        
        if not channels_index:
            st.error(f"No channel index found in the uploaded package. Please ensure it's a valid Discord export containing 'messages/index.json' or 'Mensajes/index.json'.")
            st.info(f"Searched in: {st.session_state.data_path}")
            return
            
        # Update messages_dir to be the specialized one (handling 'messages' vs 'Mensajes' subfolder)
        final_data_path = messages_dir

        # Load Stats (Cached)
        with st.spinner("Loading message statistics..."):
            df_stats = get_channel_stats(channels_index, final_data_path)
        
        # Render appropriate view based on session state
        if st.session_state.selected_channel is None:
            render_dashboard(df_stats, channels_index, final_data_path)
        else:
            render_channel_view(st.session_state.selected_channel, df_stats, final_data_path)
            
    else:
        # Welcome / Instructions Screen
        st.title("👋 Welcome to Discord Data Viewer")
        st.markdown("""
        ### How to view your data:
        1.  **Request your Data**: Go to Discord Settings -> Privacy & Safety -> Request all of my Data.
        2.  **Download**: Wait for the email and download the package (ZIP file).
        3.  **Upload**: Drag and drop the ZIP file into the sidebar 👈.
        
        **🔒 Privacy Note:**
        *   All processing happens **locally** on your computer.
        *   Your data is **never** uploaded to any external server.
        *   Temporary files are deleted when you restart the app or clear the cache.
        """)

if __name__ == "__main__":
    main()
