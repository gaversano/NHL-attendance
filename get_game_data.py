# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import requests
import json
import numpy as np
import pandas as pd

#JSON data structure
data = []

#csv data structure
df = pd.DataFrame(columns=["date", "visitor", "home", "visitor_goals", "home_goals", "attendance"])


for season in range(2015,2019+1):

    #Get request for web page
    page = requests.get("https://www.hockey-reference.com/leagues/NHL_"+str(season)+"_games.html")
    
    #Check webpage has successfully connected
    assert page.status_code == 200
    
    soup = BeautifulSoup(page.content, 'html.parser')  

    seasons = []

    
    #Parse the webpage content based on inspection of html structure
    #Date of game
    date = soup.select('tr th a')
    dates = [d.string for d in date]
    #Vistor team
    visitor_team = soup.find_all('td', attrs={"data-stat": "visitor_team_name"})
    visitors = [team.string for team in visitor_team]
    #Home team
    home_team = soup.find_all('td', attrs={"data-stat": "home_team_name"})
    homes = [team.string for team in home_team]
    #Visitor goals scored
    visitor_goal = soup.find_all('td', attrs={"data-stat": "visitor_goals"})
    visitor_goals = [team.string for team in visitor_goal]
    #Home goals scored
    home_goal = soup.find_all('td', attrs={"data-stat": "home_goals"})
    home_goals = [team.string for team in home_goal]
    #Fans in attendance
    attendance = soup.find_all('td', attrs={"data-stat": "attendance"})
    attendances = [team.string.replace(",","") for team in attendance]
    
    seasons.extend([season for s in range(len(dates))])
    
    #check size of lists are equal 
    assert len(dates) == len(visitor_team) == len(home_team) == len(visitor_goals) == len(home_goals) == len(attendances)
    
    
    #Populate the JSON semi-structured data
    for date, v, h, v_goals, h_goals, attend, seas in zip(dates, visitors, homes, visitor_goals, home_goals, attendances, seasons):
        temp_game = {}
        
        
        temp_game.update({"season" : seas})
        
        temp_game.update({"date" : date})
        
        temp_game.update({"teams" : {"home": h, "visitor": v}})
        
        temp_game.update({"score" : {"home": int(h_goals), "visitor": int(v_goals)}})
   
        
        temp_game.update({"attendence" : int(attend)})
            
        data.append(temp_game)  
        
    #Populate csv tabular file   
    temp_df = pd.DataFrame(np.column_stack([seasons, dates, visitors, homes, visitor_goals, home_goals,attendances]))
    temp_df.columns = ["season", "date", "visitor", "home", "visitor_goals", "home_goals", "attendance"]
    df = pd.concat([df, temp_df], ignore_index=True)


#Write to JSON file            
with open('game_data.json', 'w') as outfile:
    json.dump(data, outfile, indent=2)


#Write csv file
df.to_csv("game_data.txt")