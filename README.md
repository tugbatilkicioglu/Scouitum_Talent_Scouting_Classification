#Scouitum_Talent_Scouting_Classification


# TALENT SCOUTING CLASSIFICATION WITH MACHINE LEARNING


# Business Problem:

#Predicting which class (average, highlighted) a player belongs to based on the ratings given to players' attributes by scouts.


# Dataset Story:


#The dataset consists of ratings given by scouts to football players based on their attributes observed during matches, including the attributes scored during the match and their ratings.
#attributes: Contains ratings given by users evaluating players for each player observed in a match. (independent variables)
#potential_labels: Contains potential labels indicating the final opinions of users evaluating players in each match. (dependent variable)
#9 Variables, 10730 Observations, 0.65 mb


# Variables:


#task_response_id: A set of evaluations of all players in a team by a scout in a match.

#match_id: The id of the relevant match.

#evaluator_id: The id of the evaluator (scout).

#player_id: The id of the relevant player.

#position_id: The id of the position played by the relevant player in that match.

#1- Goalkeeper
#2- Centre Back
#3- Right Back
#4- Left Back
#5- Defensive Midfielder
#6- Central Midfielder
#7- Right Winger
#8- Left Winger
#9- Attacking Midfielder
#10- Forward

#analysis_id: A set containing a scout's feature evaluations of a player in a match.

#attribute_id: The id of each feature evaluated for players.

#attribute_value: The value (score) given by a scout to a player's feature.

#potential_label: A label representing the final decision of a scout regarding a player in a match. (dependent variable)


