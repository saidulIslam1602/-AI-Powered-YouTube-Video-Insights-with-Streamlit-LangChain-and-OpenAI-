"""Client dashboard for customer-facing features."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd

from src.models.client_manager import ClientManager, Client
from src.utils.logger import logger


class ClientDashboard:
    """Client dashboard for analytics and reporting."""
    
    def __init__(self, client_manager: ClientManager):
        """Initialize client dashboard."""
        self.client_manager = client_manager
    
    def render_dashboard(self, client: Client):
        """Render the main client dashboard."""
        st.markdown(f"# 🏢 {client.company_name} Dashboard")
        st.markdown(f"Welcome back, {client.contact_name}!")
        
        # Client info sidebar
        self._render_client_info_sidebar(client)
        
        # Main dashboard content
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self._render_usage_metrics(client)
        
        with col2:
            self._render_quota_status(client)
        
        with col3:
            self._render_subscription_info(client)
        
        with col4:
            self._render_last_activity(client)
        
        # Analytics section
        st.markdown("## 📊 Analytics Overview")
        
        # Time period selector
        col1, col2 = st.columns([1, 3])
        with col1:
            period = st.selectbox("Time Period", [7, 30, 90], index=1)
        with col2:
            st.markdown(f"*Showing data for the last {period} days*")
        
        # Get analytics data
        analytics = self.client_manager.get_client_analytics(client.client_id, period)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_usage_chart(analytics)
        
        with col2:
            self._render_performance_chart(analytics)
        
        # Detailed analytics
        self._render_detailed_analytics(analytics)
        
        # Feedback section
        self._render_feedback_section(client)
    
    def _render_client_info_sidebar(self, client: Client):
        """Render client information in sidebar."""
        with st.sidebar:
            st.markdown("### 🏢 Client Information")
            st.markdown(f"**Company:** {client.company_name}")
            st.markdown(f"**Contact:** {client.contact_name}")
            st.markdown(f"**Email:** {client.contact_email}")
            st.markdown(f"**Tier:** {client.subscription_tier.title()}")
            
            if client.last_login:
                st.markdown(f"**Last Login:** {client.last_login.strftime('%Y-%m-%d %H:%M')}")
            
            st.markdown("---")
            
            # Quick actions
            st.markdown("### ⚡ Quick Actions")
            if st.button("📊 Generate Report"):
                self._generate_client_report(client)
            
            if st.button("🔄 Refresh Data"):
                st.rerun()
    
    def _render_usage_metrics(self, client: Client):
        """Render usage metrics card."""
        st.markdown("### 📈 Usage Metrics")
        
        analytics = self.client_manager.get_client_analytics(client.client_id, 30)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Queries", analytics['total_queries'])
        with col2:
            st.metric("Unique Videos", analytics['unique_videos'])
        
        st.metric("Avg Response Time", f"{analytics['avg_response_time']:.2f}s")
    
    def _render_quota_status(self, client: Client):
        """Render quota status card."""
        st.markdown("### 🎯 API Quota")
        
        usage_percentage = (client.api_usage / client.api_quota) * 100
        
        st.metric("Usage", f"{client.api_usage}/{client.api_quota}")
        
        # Progress bar
        st.progress(usage_percentage / 100)
        
        if usage_percentage > 80:
            st.warning("⚠️ Quota nearly exceeded")
        elif usage_percentage > 100:
            st.error("❌ Quota exceeded")
        else:
            st.success("✅ Quota healthy")
    
    def _render_subscription_info(self, client: Client):
        """Render subscription information card."""
        st.markdown("### 💎 Subscription")
        
        tier_colors = {
            "basic": "🔵",
            "professional": "🟢", 
            "enterprise": "🟣"
        }
        
        st.markdown(f"**Tier:** {tier_colors.get(client.subscription_tier, '⚪')} {client.subscription_tier.title()}")
        st.markdown(f"**Created:** {client.created_at.strftime('%Y-%m-%d')}")
        
        if client.subscription_tier == "basic":
            st.info("💡 Upgrade to Professional for more features!")
    
    def _render_last_activity(self, client: Client):
        """Render last activity card."""
        st.markdown("### 🕒 Last Activity")
        
        if client.last_login:
            time_diff = datetime.now() - client.last_login
            if time_diff.days > 0:
                st.markdown(f"**{time_diff.days} days ago**")
            else:
                hours = time_diff.seconds // 3600
                st.markdown(f"**{hours} hours ago**")
        else:
            st.markdown("**No recent activity**")
    
    def _render_usage_chart(self, analytics: Dict[str, Any]):
        """Render usage chart."""
        st.markdown("#### 📈 Daily Usage")
        
        if analytics['daily_usage']:
            df = pd.DataFrame(analytics['daily_usage'])
            df['date'] = pd.to_datetime(df['date'])
            
            fig = px.line(df, x='date', y='queries', 
                         title="Queries per Day",
                         labels={'queries': 'Number of Queries', 'date': 'Date'})
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No usage data available for the selected period.")
    
    def _render_performance_chart(self, analytics: Dict[str, Any]):
        """Render performance chart."""
        st.markdown("#### ⚡ Performance Metrics")
        
        if analytics['total_queries'] > 0:
            # Create performance metrics
            metrics = {
                'Metric': ['Avg Response Time', 'Avg Confidence', 'Total Queries'],
                'Value': [
                    f"{analytics['avg_response_time']:.2f}s",
                    f"{analytics['avg_confidence']:.1%}",
                    analytics['total_queries']
                ]
            }
            
            fig = go.Figure(data=[
                go.Bar(x=metrics['Metric'], y=metrics['Value'], 
                      marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ])
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance data available.")
    
    def _render_detailed_analytics(self, analytics: Dict[str, Any]):
        """Render detailed analytics section."""
        st.markdown("## 📋 Detailed Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Summary Statistics")
            st.json({
                "Total Queries": analytics['total_queries'],
                "Average Response Time": f"{analytics['avg_response_time']:.2f} seconds",
                "Average Confidence": f"{analytics['avg_confidence']:.1%}",
                "Unique Videos Analyzed": analytics['unique_videos']
            })
        
        with col2:
            st.markdown("### 📈 Performance Trends")
            if analytics['daily_usage']:
                df = pd.DataFrame(analytics['daily_usage'])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No daily usage data available.")
    
    def _render_feedback_section(self, client: Client):
        """Render feedback submission section."""
        st.markdown("## 💬 Feedback & Support")
        
        with st.expander("Submit Feedback"):
            col1, col2 = st.columns(2)
            
            with col1:
                rating = st.slider("Rate your experience", 1, 5, 5)
            
            with col2:
                st.markdown("**Rating Scale:**")
                st.markdown("1 ⭐ - Poor")
                st.markdown("2 ⭐⭐ - Fair") 
                st.markdown("3 ⭐⭐⭐ - Good")
                st.markdown("4 ⭐⭐⭐⭐ - Very Good")
                st.markdown("5 ⭐⭐⭐⭐⭐ - Excellent")
            
            feedback_text = st.text_area(
                "Additional feedback (optional)",
                placeholder="Tell us how we can improve your experience..."
            )
            
            if st.button("Submit Feedback"):
                session_id = st.session_state.get('client_session_id', 'unknown')
                self.client_manager.submit_feedback(
                    client.client_id, session_id, rating, feedback_text
                )
                st.success("Thank you for your feedback! 🙏")
    
    def _generate_client_report(self, client: Client):
        """Generate a client report."""
        st.markdown("## 📄 Client Report")
        
        analytics = self.client_manager.get_client_analytics(client.client_id, 30)
        
        report_data = {
            "Client Information": {
                "Company Name": client.company_name,
                "Contact Name": client.contact_name,
                "Subscription Tier": client.subscription_tier,
                "Account Created": client.created_at.strftime('%Y-%m-%d'),
                "Last Login": client.last_login.strftime('%Y-%m-%d %H:%M') if client.last_login else "Never"
            },
            "Usage Statistics (Last 30 Days)": {
                "Total Queries": analytics['total_queries'],
                "Unique Videos": analytics['unique_videos'],
                "Average Response Time": f"{analytics['avg_response_time']:.2f} seconds",
                "Average Confidence Score": f"{analytics['avg_confidence']:.1%}",
                "API Usage": f"{client.api_usage}/{client.api_quota} ({client.api_usage/client.api_quota*100:.1f}%)"
            }
        }
        
        for section, data in report_data.items():
            st.markdown(f"### {section}")
            for key, value in data.items():
                st.markdown(f"**{key}:** {value}")
        
        # Download button for report
        if st.button("📥 Download Report"):
            st.info("Report download feature coming soon!")
    
    def render_client_onboarding(self):
        """Render client onboarding flow."""
        st.markdown("# 🚀 Welcome to Video Insights Platform")
        st.markdown("Get started by creating your account or signing in.")
        
        tab1, tab2 = st.tabs(["🔐 Sign In", "📝 Create Account"])
        
        with tab1:
            self._render_signin_form()
        
        with tab2:
            self._render_signup_form()
    
    def _render_signin_form(self):
        """Render sign-in form."""
        with st.form("signin_form"):
            st.markdown("### Sign In to Your Account")
            
            email = st.text_input("Email Address", placeholder="your@company.com")
            password = st.text_input("Password", type="password")
            
            col1, col2 = st.columns(2)
            with col1:
                signin_button = st.form_submit_button("Sign In", type="primary")
            with col2:
                if st.form_submit_button("Forgot Password?"):
                    st.info("Password reset feature coming soon!")
            
            if signin_button:
                if email and password:
                    client = self.client_manager.authenticate_client(email, password)
                    if client:
                        # Create session
                        session_id = self.client_manager.create_session(client.client_id)
                        st.session_state['client'] = client
                        st.session_state['client_session_id'] = session_id
                        st.session_state['authenticated'] = True
                        st.success("Successfully signed in!")
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")
                else:
                    st.error("Please fill in all fields.")
    
    def _render_signup_form(self):
        """Render sign-up form."""
        with st.form("signup_form"):
            st.markdown("### Create Your Account")
            
            company_name = st.text_input("Company Name", placeholder="Your Company Inc.")
            contact_name = st.text_input("Contact Name", placeholder="John Doe")
            contact_email = st.text_input("Email Address", placeholder="john@company.com")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            subscription_tier = st.selectbox("Subscription Tier", ["basic", "professional", "enterprise"])
            
            if st.form_submit_button("Create Account", type="primary"):
                if all([company_name, contact_name, contact_email, password, confirm_password]):
                    if password == confirm_password:
                        try:
                            client_id = self.client_manager.create_client(
                                company_name, contact_email, contact_name, password, subscription_tier
                            )
                            st.success("Account created successfully! Please sign in.")
                        except ValueError as e:
                            st.error(str(e))
                    else:
                        st.error("Passwords do not match.")
                else:
                    st.error("Please fill in all fields.")