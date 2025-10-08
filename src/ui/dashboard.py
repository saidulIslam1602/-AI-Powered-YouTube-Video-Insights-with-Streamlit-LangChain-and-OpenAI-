"""Dashboard and settings functionality for the Streamlit application."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from src.config.settings import settings
from src.utils.logger import logger
from src.client.client_manager import ClientManager
from src.models.client import Client


def render_client_dashboard_tab() -> None:
    """Render the client dashboard tab with analytics and usage data."""
    if not st.session_state.authenticated or not st.session_state.client:
        st.warning("Please authenticate to access the dashboard.")
        return
    
    st.markdown("## Client Dashboard")
    
    # Get client data
    client = st.session_state.client
    client_manager = st.session_state.client_manager
    
    # Dashboard header with key metrics
    render_dashboard_header(client)
    
    # Main dashboard content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_usage_analytics(client, client_manager)
        render_recent_activity(client, client_manager)
    
    with col2:
        render_account_summary(client)
        render_quick_actions(client)


def render_dashboard_header(client: Client) -> None:
    """Render the dashboard header with key metrics."""
    st.markdown(f"### Welcome back, {client.contact_name}")
    st.markdown(f"**{client.company_name}** - {client.subscription_tier.title()} Plan")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        usage_percentage = (client.api_usage / client.api_quota) * 100
        st.metric("API Usage", f"{client.api_usage}/{client.api_quota}", f"{usage_percentage:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        remaining_quota = client.api_quota - client.api_usage
        st.metric("Remaining Quota", remaining_quota)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        days_since_created = (datetime.utcnow() - client.created_at).days
        st.metric("Account Age", f"{days_since_created} days")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        status = "Active" if client.is_active else "Inactive"
        st.metric("Account Status", status)
        st.markdown('</div>', unsafe_allow_html=True)


def render_usage_analytics(client: Client, client_manager: ClientManager) -> None:
    """Render usage analytics charts and data."""
    st.markdown("### Usage Analytics")
    
    try:
        # Get usage data from client manager
        usage_data = client_manager.get_client_analytics(client.client_id)
        
        if usage_data and len(usage_data) > 0:
            # Convert to DataFrame
            df = pd.DataFrame(usage_data)
            df['date'] = pd.to_datetime(df['date'])
            
            # Usage over time chart
            fig = px.line(
                df, 
                x='date', 
                y='api_calls', 
                title='API Usage Over Time',
                color_discrete_sequence=['#0078d4']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_family="Segoe UI"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Usage by hour heatmap (if available)
            if 'hour' in df.columns:
                hourly_usage = df.groupby('hour')['api_calls'].sum().reset_index()
                fig_bar = px.bar(
                    hourly_usage,
                    x='hour',
                    y='api_calls',
                    title='Usage by Hour of Day',
                    color_discrete_sequence=['#107c10']
                )
                fig_bar.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_family="Segoe UI"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        else:
            st.info("No usage data available yet. Start analyzing videos to see your analytics!")
    
    except Exception as e:
        logger.error(f"Error rendering usage analytics: {str(e)}")
        st.error("Unable to load usage analytics. Please try again later.")


def render_recent_activity(client: Client, client_manager: ClientManager) -> None:
    """Render recent activity log."""
    st.markdown("### Recent Activity")
    
    try:
        # Get recent activity
        activities = client_manager.get_client_activity(client.client_id, limit=10)
        
        if activities:
            # Create activity DataFrame
            activity_df = pd.DataFrame(activities)
            
            # Format the activity log
            for i, activity in enumerate(activities):
                with st.expander(f"{activity.get('action', 'Unknown')} - {activity.get('timestamp', 'Unknown time')}"):
                    st.write(f"**Action:** {activity.get('action', 'N/A')}")
                    st.write(f"**Timestamp:** {activity.get('timestamp', 'N/A')}")
                    st.write(f"**Details:** {activity.get('details', 'No details available')}")
                    
                    if activity.get('video_id'):
                        st.write(f"**Video ID:** {activity['video_id']}")
                    
                    if activity.get('query'):
                        st.write(f"**Query:** {activity['query'][:100]}...")
        else:
            st.info("No recent activity found.")
    
    except Exception as e:
        logger.error(f"Error rendering recent activity: {str(e)}")
        st.error("Unable to load recent activity. Please try again later.")


def render_account_summary(client: Client) -> None:
    """Render account summary information."""
    st.markdown("### Account Summary")
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    
    # Account details
    st.markdown("**Account Information**")
    st.write(f"Client ID: `{client.client_id}`")
    st.write(f"Company: {client.company_name}")
    st.write(f"Contact: {client.contact_name}")
    st.write(f"Plan: {client.subscription_tier.title()}")
    st.write(f"Created: {client.created_at.strftime('%Y-%m-%d')}")
    
    # Quota information
    st.markdown("**Quota Details**")
    usage_percentage = (client.api_usage / client.api_quota) * 100
    
    if usage_percentage > 90:
        st.error(f"Quota almost exhausted: {usage_percentage:.1f}%")
    elif usage_percentage > 75:
        st.warning(f"High usage: {usage_percentage:.1f}%")
    else:
        st.success(f"Usage: {usage_percentage:.1f}%")
    
    # Progress bar
    st.progress(usage_percentage / 100)
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_quick_actions(client: Client) -> None:
    """Render quick action buttons."""
    st.markdown("### Quick Actions")
    
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    
    # Quick action buttons
    if st.button("Clear Cache", type="secondary"):
        if st.session_state.video_processor:
            st.session_state.video_processor.clear_cache()
            st.success("Cache cleared successfully!")
            st.rerun()
    
    if st.button("Export Usage Data", type="secondary"):
        try:
            # Generate usage report
            usage_data = st.session_state.client_manager.get_client_analytics(client.client_id)
            if usage_data:
                df = pd.DataFrame(usage_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"usage_report_{client.client_id}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No usage data available for export.")
        except Exception as e:
            st.error("Failed to export usage data.")
            logger.error(f"Export error: {str(e)}")
    
    if st.button("Reset Password", type="secondary"):
        st.info("Password reset functionality will be available soon.")
    
    st.markdown('</div>', unsafe_allow_html=True)


def render_settings_tab() -> None:
    """Render the settings tab with application and account settings."""
    if not st.session_state.authenticated or not st.session_state.client:
        st.warning("Please authenticate to access settings.")
        return
    
    st.markdown("## Settings")
    
    # Settings categories
    tab1, tab2, tab3 = st.tabs(["Application Settings", "Account Settings", "Advanced"])
    
    with tab1:
        render_application_settings()
    
    with tab2:
        render_account_settings()
    
    with tab3:
        render_advanced_settings()


def render_application_settings() -> None:
    """Render application-specific settings."""
    st.markdown("### Application Settings")
    
    # Video processing settings
    st.markdown("#### Video Processing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        chunk_size = st.slider(
            "Text Chunk Size",
            min_value=100,
            max_value=2000,
            value=settings.chunk_size,
            step=100,
            help="Size of text chunks for processing"
        )
    
    with col2:
        max_video_length = st.slider(
            "Max Video Length (minutes)",
            min_value=10,
            max_value=180,
            value=settings.max_video_length_minutes,
            step=10,
            help="Maximum allowed video length"
        )
    
    # Language settings
    st.markdown("#### Language Settings")
    preferred_language = st.selectbox(
        "Preferred Language",
        ["auto", "en", "es", "fr", "de", "it", "pt", "ja", "ko", "zh"],
        index=0,
        help="Preferred language for video processing"
    )
    
    # Cache settings
    st.markdown("#### Cache Settings")
    enable_cache = st.checkbox("Enable Caching", value=True, help="Cache processed videos for faster access")
    
    if enable_cache:
        cache_ttl = st.slider(
            "Cache TTL (hours)",
            min_value=1,
            max_value=168,
            value=24,
            help="How long to keep cached data"
        )
    
    # Save settings
    if st.button("Save Application Settings", type="primary"):
        # Here you would save the settings to the database or session state
        st.success("Application settings saved successfully!")


def render_account_settings() -> None:
    """Render account-specific settings."""
    st.markdown("### Account Settings")
    
    client = st.session_state.client
    
    # Account information
    st.markdown("#### Account Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_company_name = st.text_input("Company Name", value=client.company_name)
        new_contact_name = st.text_input("Contact Name", value=client.contact_name)
    
    with col2:
        # Display read-only information
        st.text_input("Client ID", value=client.client_id, disabled=True)
        st.text_input("Subscription Tier", value=client.subscription_tier.title(), disabled=True)
    
    # Notification settings
    st.markdown("#### Notification Settings")
    
    email_notifications = st.checkbox("Email Notifications", value=True)
    quota_warnings = st.checkbox("Quota Warning Alerts", value=True)
    security_alerts = st.checkbox("Security Alerts", value=True)
    
    # Privacy settings
    st.markdown("#### Privacy Settings")
    
    data_retention = st.selectbox(
        "Data Retention Period",
        ["30 days", "90 days", "1 year", "2 years"],
        index=2
    )
    
    allow_analytics = st.checkbox("Allow Usage Analytics", value=True)
    
    # Save account settings
    if st.button("Save Account Settings", type="primary"):
        try:
            # Update client information
            st.session_state.client_manager.update_client(
                client.client_id,
                company_name=new_company_name,
                contact_name=new_contact_name
            )
            
            # Update session state
            st.session_state.client.company_name = new_company_name
            st.session_state.client.contact_name = new_contact_name
            
            st.success("Account settings saved successfully!")
            st.rerun()
        
        except Exception as e:
            st.error("Failed to save account settings.")
            logger.error(f"Account settings update error: {str(e)}")


def render_advanced_settings() -> None:
    """Render advanced settings for power users."""
    st.markdown("### Advanced Settings")
    
    st.warning("⚠️ Advanced settings should only be modified by experienced users.")
    
    # API settings
    st.markdown("#### API Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        api_timeout = st.number_input(
            "API Timeout (seconds)",
            min_value=10,
            max_value=300,
            value=60,
            help="Timeout for API requests"
        )
    
    with col2:
        max_retries = st.number_input(
            "Max Retries",
            min_value=1,
            max_value=10,
            value=3,
            help="Maximum number of retry attempts"
        )
    
    # Debug settings
    st.markdown("#### Debug Settings")
    
    enable_debug = st.checkbox("Enable Debug Mode", value=settings.debug)
    verbose_logging = st.checkbox("Verbose Logging", value=False)
    
    # Performance settings
    st.markdown("#### Performance Settings")
    
    concurrent_requests = st.slider(
        "Concurrent Requests",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of concurrent API requests"
    )
    
    # Data management
    st.markdown("#### Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Clear All Cache", type="secondary"):
            if st.session_state.video_processor:
                st.session_state.video_processor.clear_cache()
                st.success("All cache cleared!")
    
    with col2:
        if st.button("Reset Statistics", type="secondary"):
            st.warning("This action cannot be undone!")
    
    with col3:
        if st.button("Export All Data", type="secondary"):
            st.info("Data export feature coming soon!")
    
    # Save advanced settings
    if st.button("Save Advanced Settings", type="primary"):
        st.success("Advanced settings saved successfully!")
        st.info("Some settings may require application restart to take effect.")


def render_admin_panel() -> None:
    """Render admin panel for system administrators."""
    if not st.session_state.authenticated or not st.session_state.client:
        st.error("Authentication required.")
        return
    
    # Check if user has admin permissions
    if 'admin' not in st.session_state.user_permissions:
        st.error("Admin access required.")
        return
    
    st.markdown("## Admin Panel")
    
    # Admin functionality tabs
    tab1, tab2, tab3, tab4 = st.tabs(["User Management", "System Stats", "Logs", "Configuration"])
    
    with tab1:
        render_user_management()
    
    with tab2:
        render_system_statistics()
    
    with tab3:
        render_system_logs()
    
    with tab4:
        render_system_configuration()


def render_user_management() -> None:
    """Render user management interface for admins."""
    st.markdown("### User Management")
    
    try:
        # Get all clients
        all_clients = st.session_state.client_manager.get_all_clients()
        
        if all_clients:
            # Display clients in a table
            client_data = []
            for client in all_clients:
                client_data.append({
                    'Client ID': client.client_id,
                    'Company': client.company_name,
                    'Contact': client.contact_name,
                    'Tier': client.subscription_tier,
                    'Usage': f"{client.api_usage}/{client.api_quota}",
                    'Status': 'Active' if client.is_active else 'Inactive',
                    'Created': client.created_at.strftime('%Y-%m-%d')
                })
            
            df = pd.DataFrame(client_data)
            st.dataframe(df, use_container_width=True)
            
            # Client management actions
            st.markdown("#### Client Actions")
            
            selected_client = st.selectbox("Select Client", [c.client_id for c in all_clients])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("Reset Quota"):
                    # Reset quota logic here
                    st.success(f"Quota reset for {selected_client}")
            
            with col2:
                if st.button("Suspend Client"):
                    # Suspend client logic here
                    st.warning(f"Client {selected_client} suspended")
            
            with col3:
                if st.button("Activate Client"):
                    # Activate client logic here
                    st.success(f"Client {selected_client} activated")
            
            with col4:
                if st.button("Delete Client"):
                    st.error("Client deletion requires additional confirmation")
        
        else:
            st.info("No clients found.")
    
    except Exception as e:
        st.error("Failed to load user management data.")
        logger.error(f"User management error: {str(e)}")


def render_system_statistics() -> None:
    """Render system-wide statistics."""
    st.markdown("### System Statistics")
    
    # Mock system stats - replace with real data
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Clients", "127", "+5")
    
    with col2:
        st.metric("Videos Processed", "1,234", "+23")
    
    with col3:
        st.metric("Total Queries", "5,678", "+89")
    
    with col4:
        st.metric("System Uptime", "99.9%", "0%")
    
    # System performance chart
    st.markdown("#### System Performance")
    
    # Mock performance data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
    performance_data = {
        'Date': dates,
        'API Calls': [100 + i * 5 + (i % 7) * 20 for i in range(len(dates))],
        'Response Time': [0.5 + (i % 5) * 0.1 for i in range(len(dates))]
    }
    
    df = pd.DataFrame(performance_data)
    
    fig = px.line(df, x='Date', y='API Calls', title='Daily API Calls')
    st.plotly_chart(fig, use_container_width=True)


def render_system_logs() -> None:
    """Render system logs."""
    st.markdown("### System Logs")
    
    # Log level filter
    log_level = st.selectbox("Log Level", ["ALL", "ERROR", "WARNING", "INFO", "DEBUG"])
    
    # Mock log entries
    log_entries = [
        "2024-01-15 10:30:25 INFO - Client authenticated: client_123",
        "2024-01-15 10:30:30 INFO - Video processed: dQw4w9WgXcQ",
        "2024-01-15 10:30:35 WARNING - High API usage for client_456",
        "2024-01-15 10:30:40 ERROR - Failed to process video: invalid URL",
        "2024-01-15 10:30:45 INFO - Cache cleared by admin",
    ]
    
    # Display logs
    for entry in log_entries:
        if log_level == "ALL" or log_level in entry:
            if "ERROR" in entry:
                st.error(entry)
            elif "WARNING" in entry:
                st.warning(entry)
            else:
                st.info(entry)


def render_system_configuration() -> None:
    """Render system configuration interface."""
    st.markdown("### System Configuration")
    
    st.warning("⚠️ System configuration changes affect all users.")
    
    # Configuration sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### API Configuration")
        
        global_rate_limit = st.number_input("Global Rate Limit (requests/min)", value=1000)
        default_quota = st.number_input("Default Client Quota", value=100)
        
        st.markdown("#### Security Settings")
        
        session_timeout = st.number_input("Session Timeout (hours)", value=24)
        require_2fa = st.checkbox("Require Two-Factor Authentication")
    
    with col2:
        st.markdown("#### Performance Settings")
        
        cache_size = st.number_input("Cache Size (MB)", value=1024)
        worker_processes = st.number_input("Worker Processes", value=4)
        
        st.markdown("#### Maintenance")
        
        maintenance_mode = st.checkbox("Maintenance Mode")
        backup_enabled = st.checkbox("Automatic Backups", value=True)
    
    # Save configuration
    if st.button("Save Configuration", type="primary"):
        st.success("System configuration saved successfully!")
        st.info("Configuration changes will take effect after system restart.")