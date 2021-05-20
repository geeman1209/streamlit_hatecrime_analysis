# -*- coding: utf-8 -*-
"""
Created on Fri May 14 20:09:26 2021

@author: gabre
"""

import streamlit as st
import pandas as pd
import geopy.geocoders
from geopy.geocoders import Nominatim
import numpy as np
import plotly
import matplotlib as plt
import plotly.express as px
geopy.geocoders.options.default_user_agent = "FinalProject"


st.set_page_config(layout="wide")


st.title("Data 608 Final Project: Hate Crime Analysis using Streamlit" )

st.subheader("Introduction")

st.markdown("""
            The Trump administration in many ways was a polarizing administration. The stark divide between political
            parties and their ideologies became very apparent during Trump’s presidency. The Trump administration
            saw the rise of the BLM movement, white nationalism, ultra right wing conspiracy theories, racial violence,
            protests, etc. The purpose of my final project is to investigate the recorded hate crimes and see if there was
            a rise in reported hate crimes during the Trump presidency.
            """)

st.subheader("The Data Collection and Cleaning Process")
st.markdown("""Analyzing hate crimes on a national scale from 2010 to 2019 is not an easy task for a variety of reasons. 
            The data set is large and requires cleaning, transformation, and elimination of extraneous columns, in an effort to reduce the size and optimize loading times. 
            
            In order to create and condense some of the visualizations columns needed transformation; 
            target_loc, created for the purpose of using geopandas to find the coordinates, latitude and longitude, 
            bias_desc (variables were consolidated), and offense_name(variables were consolidated).
            """)

st.markdown(""" The dataset below is the final result after all the necessary data cleaning:  """)



###############################################################################
######Retrieve Data and Cleanup###########

#Create a function that wil return the dataframe
@st.cache
def get_data():
    dataFrame = pd.read_csv('./hate_crime.csv')
    return dataFrame

def split_name(row):
    row['OFFENSE_NAME'] = row['OFFENSE_NAME'].split(' ')
    if(row['OFFENSE_NAME'][0] == 'Aggravated' or row['OFFENSE_NAME'][0] == 'Simple'):
        row['OFFENSE_NAME'] = 'Assault'
    elif(row['OFFENSE_NAME'][0] == 'Murder'):
        row['OFFENSE_NAME'] = 'Murder'
    elif(row['OFFENSE_NAME'][0] == 'Intimidation'):
        row['OFFENSE_NAME'] = 'Intimidation'
    elif(row['OFFENSE_NAME'][0] == 'Destruction/Damage/Vandalism' or row['OFFENSE_NAME'][0] == 'Arson'):
        row['OFFENSE_NAME'] = 'Property Damage'
    elif(row['OFFENSE_NAME'][0] == 'Motor' or row['OFFENSE_NAME'][0] == 'Theft'):
        row['OFFENSE_NAME'] = 'Theft/Burglary'
    elif(row['OFFENSE_NAME'][0] == 'Burglary/Breaking' or row['OFFENSE_NAME'][0] == 'Shoplifting'):
        row['OFFENSE_NAME'] = 'Theft/Burglary'
    else:
        row['OFFENSE_NAME'] = 'Other'
    return row

def split_bias(row):
    row['BIAS_DESC'] = row['BIAS_DESC'].split(' ')
    if(row['BIAS_DESC'][0] == 'Anti-Other' or row['BIAS_DESC'][0] == 'Anti-Multiple'):
        row['BIAS_DESC'] = 'Anti-Other/Multiple Races'
    elif(row['BIAS_DESC'][0] == 'Anti-Black'):
        row['BIAS_DESC'] = 'Anti-Black'
    elif(row['BIAS_DESC'][0] == 'Anti-Hispanic'):
        row['BIAS_DESC'] = 'Anti-Latino'
    elif(row['BIAS_DESC'][0] == 'Anti-Asian'):
        row['BIAS_DESC'] = 'Anti-Asian'
    elif(row['BIAS_DESC'][0] == 'Anti-White'):
        row['BIAS_DESC'] = 'Anti-White'
    elif(row['BIAS_DESC'][0] == 'Anti-Gay'):
        row['BIAS_DESC'] = 'Anti-Gay'
    elif(row['BIAS_DESC'][0] == 'Anti-Islamic'):
        row['BIAS_DESC'] = 'Anti-Islamic'
    elif(row['BIAS_DESC'][0] == 'Anti-Arab'):
        row['BIAS_DESC'] = 'Anti-Arab'
    elif(row['BIAS_DESC'][0] == 'Anti-Jewish'):
        row['BIAS_DESC'] = 'Anti-Jewish'
    elif(row['BIAS_DESC'][0] == 'Anti-White'):
        row['BIAS_DESC'] = 'Anti-White'
    elif(row['BIAS_DESC'][0] == 'Anti-Lesbian' or row['BIAS_DESC'][0] == 'Anti-Bisexual'):
        row['BIAS_DESC'] = 'Anti-Gender Non-Conforming'
    elif(row['BIAS_DESC'][0] == 'Anti-Mental' or row['BIAS_DESC'][0] == 'Anti-Physical'):
        row['BIAS_DESC'] = 'Anti-Mental/Physical Disability'
    elif(row['BIAS_DESC'][0] == 'Anti-Catholic' or row['BIAS_DESC'][0] == 'Anti-Protestant' or row['BIAS_DESC'][0] == 'Anti-Jehovah\'s' or row['BIAS_DESC'][0] == 'Anti-Mormon'):
        row['BIAS_DESC'] = 'Anti-Christian Religion'
    elif(row['BIAS_DESC'][0] == 'Anti-Sikh' or row['BIAS_DESC'][0] == 'Anti-Buddhist' or row['BIAS_DESC'][0] == 'Anti-Hindu'):
        row['BIAS_DESC'] = 'Anti-OtherReligion'
    else:
        row['BIAS_DESC'] = 'Other'
    return row

#Call the function to retrieve the data
df = get_data()

df = df.apply(split_name, axis = 1)
df = df.apply(split_bias, axis = 1)

#Create a function to initiate cleaning the data
#There are multiple columns that are not necessary for the planned visualizations
def clean_data(dataFrame):
    del dataFrame['INCIDENT_ID']
    del dataFrame['ORI']
    del dataFrame['PUB_AGENCY_UNIT']
    del dataFrame['AGENCY_TYPE_NAME']
    del dataFrame['POPULATION_GROUP_CODE']
    del dataFrame['POPULATION_GROUP_DESC']
    del dataFrame['INCIDENT_DATE']
    del dataFrame['ADULT_VICTIM_COUNT']
    del dataFrame['JUVENILE_VICTIM_COUNT']
    del dataFrame['ADULT_OFFENDER_COUNT']
    del dataFrame['JUVENILE_OFFENDER_COUNT']
    del dataFrame['OFFENDER_ETHNICITY']
    del dataFrame['MULTIPLE_OFFENSE']
    del dataFrame['MULTIPLE_BIAS']
    return dataFrame

df2 = clean_data(df).copy()

#Make New Column with city/town + state

df2["target_loc"] = df2['PUB_AGENCY_NAME'] + ", " +df2['STATE_NAME']

#Create a new data frame with all the unique locations
#This reduces the number of times the geolocator.geocode function is called
new_frame = df2["target_loc"].unique()
new_frame = pd.DataFrame(new_frame, columns=['target_loc'])
geolocator = Nominatim()

#Apply a rate limiter to mitigate time out requests
from geopy.extra.rate_limiter import RateLimiter
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

#Apply a progres bar
from tqdm import tqdm
tqdm.pandas()

def eval_results(x):
    try:
        return (x.latitude, x.longitude)
    except:
        return (None, None)

#Find the locations, latitudes and longitudes of all the reported crimes

#Why is the code commented? The code below takes at least 45 minutes to run.
    
#The code was run, retrieved the coordinates and split into latitude and longitude
#The output was exported as a csv and that data will be used moving forward 
    
#new_frame["Coordinates"] = new_frame["target_loc"].progress_apply(geolocator.geocode).apply(lambda x: eval_results(x)) 
#new_frame2 = new_frame.copy()
#new_frame2[["latitude","longitude"]] = pd.DataFrame(new_frame2["Coordinates"].tolist(), index=new_frame2.index)
    
def get_coord_data():
    dataFrame = pd.read_csv('./Coordinates.csv')
    return dataFrame

coordinates = get_coord_data()

#clean coordinates and remove null values
def clean_coordinates(df):
    df2 = df.loc[df['latitude'].notnull(), ['target_loc', 'Coordinates','latitude', 'longitude']]
    return df2

coordinates = clean_coordinates(coordinates)

def merge_coordinates(df1, df2):
    dfNew = pd.merge(df1, df2, on='target_loc', how='left')
    return dfNew



df3 = merge_coordinates(df2, coordinates)

dfinal = df3.dropna()

st.markdown("The clean data frame that will be used for the visualizations:")


st.dataframe(dfinal)
st.subheader("Visualizations and Explanations")

###############################################################################
############Visualizations#####################################################

colA, colB = st.beta_columns(2)
expander1 = st.beta_expander("Click for More Information", expanded=False)
with expander1:
    st.markdown("""
                The line graphs show the change over time for total hate crimes (first graph) 
                and which anti-bias increased or decreased over the years. 
                
                Th first graph shows there was an overall downward trend
                before 2014 and then a clear upward trend. It is worth noting that 
                the Trump campaign began in 2015, however that could be circumstantial.
                
                The second graph displays the increases and changes in hate crime motivation.
                It is worth noting that Anti-Latino bias saw an increase after the Trump campaign 
                and presidency.
                
                """)
######Visualization 1######

with colA:   
    def line_df(df):
        dummyvar = df["DATA_YEAR"].value_counts().to_frame()
        dummyvar.reset_index(inplace=True)
        dummyvar = dummyvar.rename(columns = {'index':'Year'})
        dummyvar = dummyvar.rename(columns={"DATA_YEAR":"Total_Crimes"})
        dummyvar = dummyvar.sort_values(by="Year")
        return dummyvar
            
    lineDf = line_df(dfinal)
            
            
    fig1 = px.line(lineDf, x="Year", y ="Total_Crimes")
            
    fig1.update_traces(mode = 'markers+lines')
    fig1.update_layout(title= "Total Hate Crmes from 2010 - 2019")
    st.plotly_chart(fig1)
      
with colB:
    def grouped_df(df):
        grouped_df = df.groupby("DATA_YEAR")
        return grouped_df
        
    g_df = grouped_df(dfinal)
    g_df = g_df["BIAS_DESC"].value_counts().to_frame()
    g_df = g_df.rename(columns={"BIAS_DESC":"BD_COUNT"})
    g_df = g_df.reset_index()
        
    fig2 = px.line(g_df, x="DATA_YEAR", y="BD_COUNT", color="BIAS_DESC")
    fig2.update_traces(mode = 'markers+lines')   
    st.plotly_chart(fig2)


st.markdown("------------------")
#######Visualization 3 & Visualization 4#########
expander2 = st.beta_expander("Click for More Information", expanded=False)
col1,col2 = st.beta_columns(2)
with col1:

    import plotly.graph_objects as go
    
    fig3 = go.Figure(data=go.Pie(labels=dfinal["BIAS_DESC"], values=dfinal["BIAS_DESC"].value_counts(), hole=.3))
    fig3.update_layout(title = "Anti-Bias Percentage of Hate Crimes from 2010-2019")
    st.plotly_chart(fig3)

with col2:
    fig4 = go.Figure(data=go.Pie(labels=dfinal["OFFENSE_NAME"].unique(), values=dfinal["OFFENSE_NAME"].value_counts(), hole=.3))
    fig4.update_layout(title = "Hate Crime Offense from 2010-2019")
    st.plotly_chart(fig4)
    

with expander2:
    st.markdown("""
                The pie graphs show the percentages of anti-bias across the years and which crime is associated 
                the most with hate crimes. It's revealing to see that majority of crimes associated with 
                hate crimes are violent.
                """)

st.markdown("----------")
####Visualization 4######
expander3 = st.beta_expander("Click for More Information", expanded=False)

px.set_mapbox_access_token("pk.eyJ1IjoiZ2VlbWFuMTIwOSIsImEiOiJja291aTI5bjAwbDZ6Mm9sbGVxb292c3NiIn0.eKZKp_JepI5fBsYRa2cdzw")


fig5 = px.scatter_mapbox(dfinal, 
                         lat="latitude", 
                         lon="longitude", 
                         animation_frame="DATA_YEAR",
                         animation_group="BIAS_DESC",
                         color="BIAS_DESC",
                         hover_name="BIAS_DESC", 
                         size="VICTIM_COUNT", 
                         hover_data=["target_loc"],
                         color_continuous_scale=px.colors.cyclical.IceFire,
                         size_max=35,
                         zoom= 3,
                         width= 1500
                         )


fig5.update_layout(
        title = 'Hate Crimes per year<br>(Hover for City,State/Bias Description/Victim Count)',
        autosize= True,
        geo_scope='usa',
    )
st.plotly_chart(fig5)

with expander3:
    st.markdown("""
                This is a scatter map using mapbox's API, showing the geogrpahic distribution of hate crimes per year.
                If you haver over the various points, you will see the location and victim count.
                """)
st.markdown("---")


######Visualization 5###########


fig6 = px.parallel_categories(dfinal, dimensions=['BIAS_DESC','DATA_YEAR', 'REGION_NAME', 'OFFENSE_NAME'], 
                color_continuous_scale=px.colors.sequential.Inferno
                )
fig6.update_layout(
        height=600,
        width=1100,
        dragmode='lasso', 
        hovermode='closest')
st.plotly_chart(fig6)


expander4 = st.beta_expander("Click for More Information", expanded=False)
with expander4:
    st.markdown("""
                The Parallel Category Diagram visualizes the relationship between the 
                categorical variables. In the case of this diagram, I selected anti-bias, data year, and offense in relation 
                to the anti-bias crimes.The rectangles shows the frequency of the variable and the ribbon connecting the various 
                rectangles show the relative frequency.
                
                There are interesting relationships to be observed. Anti-Black and Anti-Gay bias are subjected to a lot proportion of 
                assaults and intimidation. While Anti-Jewish Bias is subjected to a higher degree of property crime.
                """)

st.markdown("---------------------------------------------------------------")


######Visualization 6 & Visualization 7#########

container = st.beta_container()

with container:
    columns1,columns2 = st.beta_columns(2)
    
    year = st.selectbox('Select a year', options=dfinal['DATA_YEAR'].unique())
    bias = st.selectbox("Select a Bias", options=dfinal['BIAS_DESC'].unique())
    
    
    def bar_df(df, year, bias):
        selected_df = df[(df["DATA_YEAR"] == year) & (df["BIAS_DESC"] == bias)]
        selected_df = selected_df.filter(['REGION_NAME', 'VICTIM_TYPES', 'OFFENSE_NAME', 'VICTIM_COUNT'])
        return selected_df
    
    df_bar = bar_df(dfinal, year, bias)
    
    # df_bar = bar_df(dfinal, year, bias)
    
    with columns1:

        fig7 =px.bar(df_bar, x="REGION_NAME", y = "VICTIM_COUNT", color="VICTIM_TYPES")
        st.plotly_chart(fig7)
        
    with columns2:
        fig8 =px.bar(df_bar, x="REGION_NAME", y = "VICTIM_COUNT", color="OFFENSE_NAME", barmode="group")
        st.plotly_chart(fig8)


expander5 = st.beta_expander("Click for More Information", expanded=False)
with expander5:
    st.markdown("""
                These bar graphs take in the year and anti-bias you wish to investigate further.
                The first graph gives the victim count and region per bias while highlighting the victim types while the 
                second graph highlights the offense type typically associated with the anti-bias per region.
                """)    
