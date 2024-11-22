import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

from openai import OpenAI
import os
from dotenv import load_dotenv


class RenewableEnergyDashboard:
    def __init__(self):
        self.data = pd.read_csv('data/processed/tableau_main_data.csv')
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def create_dashboard(self):
        st.markdown(
            "<h1 style='text-align: center;'>Global Renewable Energy Analytics</h1>",
            unsafe_allow_html=True,
        )

        # Main menu for navigation
        menu = st.sidebar.selectbox(
            "Choose a functionality",
            ["Dashboard", "Predictive Intelligence", "AI Assistant"],
        )

        if menu == "Dashboard":
            self.dashboard()
        elif menu == "Predictive Intelligence":
            self.predictive_intelligence()
        elif menu == "AI Assistant":
            self.ai_powered_assistant()

    def dashboard(self):
        # Embed Tableau
        tableau_html = """
        <div class='tableauPlaceholder' id='viz1732298488749' style='position: relative; width: 100%; height: 800px;'>
            <noscript>
                <a href='#'>
                    <img alt='Global Renewable Energy Analysis: Relationships between Renewable Output, Economic Development, and Energy Access '
                         src='https://public.tableau.com/static/images/Re/RenewableEnergyAnalytics/Dashboard1/1_rss.png'
                         style='border: none' />
                </a>
            </noscript>
            <object class='tableauViz' style='display:none;'>
                <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
                <param name='embed_code_version' value='3' />
                <param name='site_root' value='' />
                <param name='name' value='RenewableEnergyAnalytics/Dashboard1' />
                <param name='tabs' value='no' />
                <param name='toolbar' value='yes' />
                <param name='static_image' value='https://public.tableau.com/static/images/Re/RenewableEnergyAnalytics/Dashboard1/1.png' />
                <param name='animate_transition' value='yes' />
                <param name='display_static_image' value='yes' />
                <param name='display_spinner' value='yes' />
                <param name='display_overlay' value='yes' />
                <param name='display_count' value='yes' />
                <param name='language' value='en-US' />
                <param name='filter' value='publish=yes' />
            </object>
        </div>
        <script type='text/javascript'>
            var divElement = document.getElementById('viz1732298488749');
            var vizElement = divElement.getElementsByTagName('object')[0];
            vizElement.style.width = '100%';  // Full-width adjustment
            vizElement.style.height = '800px';  // Adjust height as needed
            var scriptElement = document.createElement('script');
            scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
            vizElement.parentNode.insertBefore(scriptElement, vizElement);
        </script>
        """

        # Embed Tableau dashboard
        st.components.v1.html(tableau_html, height=800)

        # Interactive Analysis Section
        st.header("Interactive Analysis")

        # Country selection
        countries = st.multiselect(
            "Select Countries for Comparison",
            self.data['Country'].unique()
        )

        if countries:
            filtered_data = self.data[self.data['Country'].isin(countries)]

            # Create comparison charts
            col1, col2 = st.columns(2)

            with col1:
                fig1 = px.line(
                    filtered_data,
                    x='year',
                    y='Renewable_Electricity_Output',
                    color='Country',
                    title='Renewable Energy Adoption Trends'
                )
                st.plotly_chart(fig1)

            with col2:
                fig2 = px.scatter(
                    filtered_data,
                    x='GDP_Per_Capita',
                    y='Renewable_Electricity_Output',
                    color='Country',
                    size='EG.ELC.ACCS.ZS',
                    title='GDP vs Renewable Energy'
                )
                st.plotly_chart(fig2)


     # AI Analysis Section
        st.header("AI Analysis")
        if countries:
            query = f"Analyze renewable energy trends for {', '.join(countries)}"
            with st.spinner('Generating insights...'):
                try:
                    completion = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": query}]
                    )
                    st.write(completion.choices[0].message.content)
                except Exception as e:
                    st.error("Error fetching AI insights. Please check your API key or query.")
            

    def predictive_intelligence(self):
        st.header("Predictive Intelligence")
    
        st.subheader("Renewable Energy Growth Prediction")
        st.write("Predict future renewable energy capacity based on economic, social, and technological factors.")

        # User inputs
        gdp = st.number_input("GDP Per Capita (USD)", min_value=0, step=100, value=10000)
        energy_access = st.slider("Energy Access (% of population)", 0, 100, 80)
        tech_adoption = st.slider("Technology Adoption Rate (%)", 0, 100, 50)

        # Mock data for training
        X_train = pd.DataFrame({
            'GDP_Per_Capita': [5000, 10000, 20000],
            'Energy_Access': [60, 80, 90],
            'Technology_Adoption': [30, 50, 80],
        })
        y_train = [50, 100, 200]  # Renewable energy output in MW

        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        user_input = pd.DataFrame({
            'GDP_Per_Capita': [gdp],
            'Energy_Access': [energy_access],
            'Technology_Adoption': [tech_adoption],
        })
        prediction = model.predict(user_input)[0]

        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        # Display prediction
        st.success(f"Predicted Renewable Energy Capacity: {prediction:.2f} MW")

        # Explain factors
        st.write("**Factors Influencing the Prediction:**")
        for _, row in feature_importance.iterrows():
            st.write(f"- **{row['Feature']}**: {row['Importance']:.2f} (relative importance)")

        # Visualization
        st.subheader("Scenario Analysis")
        st.write("Adjust inputs to see how renewable energy capacity changes under different scenarios.")

        # Generate predictions for varying GDP and technology adoption rates
        gdp_range = range(5000, 25000, 5000)
        predictions = []
        for g in gdp_range:
            user_input = pd.DataFrame({
                'GDP_Per_Capita': [g],
                'Energy_Access': [energy_access],
                'Technology_Adoption': [tech_adoption],
            })
            pred = model.predict(user_input)[0]  # Ensure a scalar value is appended
            predictions.append(pred)  # Append scalar predictions

        # Plot predictions
        fig, ax = plt.subplots()
        ax.plot(gdp_range, predictions, marker='o', linestyle='-', label='Predicted Capacity')
        ax.set_title("Predicted Renewable Energy Capacity vs. GDP")
        ax.set_xlabel("GDP Per Capita (USD)")
        ax.set_ylabel("Renewable Energy Capacity (MW)")
        ax.legend()
        st.pyplot(fig)

        st.write("""
        **Insights**:
        - Higher GDP typically correlates with increased renewable energy capacity.
        - Energy access and technology adoption rates are critical multipliers for growth.
        - Policy interventions targeting technology adoption can significantly boost capacity.
        """)

    def ai_powered_assistant(self):
        st.header("AI-Powered Assistant")
        
        # Input for user queries
        user_query = st.text_area("Ask any question about the data, trends, or insights:")
        
        if user_query:
            with st.spinner('Fetching insights...'):
                try:
                    # ChatGPT API call
                    response = self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": user_query}]
                    )
                    st.write(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"Error fetching insights: {e}")


if __name__ == "__main__":
    dashboard = RenewableEnergyDashboard()
    dashboard.create_dashboard()
