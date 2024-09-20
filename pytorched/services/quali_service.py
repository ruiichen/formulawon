import requests
import pandas as pd
from dateutil.relativedelta import *
from util.consts import headers, booleans
from util.exceptions import NotFoundError, ExternalServiceError


def get_quali(season, round):
    qualifying_results = {'grid': [],
                          'driver': [],
                          'constructor': [],
                          'qualifying_time': [],
                          'season': [],
                          'round': []}

    url = 'http://ergast.com/api/f1/{}/{}/qualifying.json'
    try:
        resp = requests.get(url.format(season, round))
    except:
        raise ExternalServiceError()
    json = resp.json()
    if not json['MRData']['RaceTable']['Races']:
        raise NotFoundError()

    for item in json['MRData']['RaceTable']['Races'][0]['QualifyingResults']:
        qualifying_results['grid'].append(int(item['position']))
        qualifying_results['driver'].append(item['Driver']['driverId'])
        qualifying_results['constructor'].append(item['Constructor']['constructorId'])
        try:
            if 'Q3' in item and item['Q3']:
                qualifying_results['qualifying_time'].append(item['Q3'])
            elif 'Q2' in item and item['Q2']:
                qualifying_results['qualifying_time'].append(item['Q2'])
            elif 'Q1' in item and item['Q1']:
                qualifying_results['qualifying_time'].append(item['Q1'])
            else:
                qualifying_results['qualifying_time'].append("00.000")
        except:
            qualifying_results['qualifying_time'].append("00.000")
            qualifying_results['Q'].append(int(1))
        qualifying_results['season'].append(int(json['MRData']['RaceTable']['Races'][0]['season']))
        qualifying_results['round'].append(int(json['MRData']['RaceTable']['Races'][0]['round']))
    return  pd.DataFrame(qualifying_results)

def get_drivers(season, round):
    driver_standings = {'season': [],
                        'round': [],
                        'driver': [],
                        'driver_points': [],
                        'driver_wins': [],
                        'driver_standings_pos': [],
                        'nationality': [],
                        'date_of_birth': []}

    url = 'https://ergast.com/api/f1/{}/{}/driverStandings.json'
    try:
        resp = requests.get(url.format(season, round))
    except:
        raise ExternalServiceError()
    json = resp.json()

    for item in json['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']:
        driver_standings['season'].append(int(json['MRData']['StandingsTable']['StandingsLists'][0]['season']))
        driver_standings['round'].append(int(json['MRData']['StandingsTable']['StandingsLists'][0]['round']))
        driver_standings['driver'].append(item['Driver']['driverId'])
        driver_standings['nationality'].append(item['Driver']['nationality'])
        driver_standings['date_of_birth'].append(item['Driver']['dateOfBirth'])
        try:
            driver_standings['driver_points'].append(int(item['points']))
        except:
            driver_standings['driver_points'].append(0)
        driver_standings['driver_wins'].append(int(item['wins']))
        driver_standings['driver_standings_pos'].append(int(item['position']))
    return  pd.DataFrame(driver_standings)

def get_constructors(season, round):
    constructor_standings = {'season': [],
                             'round': [],
                             'constructor': [],
                             'constructor_points': [],
                             'constructor_wins': [],
                             'constructor_standings_pos': []}

    url = 'https://ergast.com/api/f1/{}/{}/constructorStandings.json'
    try:
        resp = requests.get(url.format(season, round))
    except:
        raise ExternalServiceError()
    json = resp.json()

    for item in json['MRData']['StandingsTable']['StandingsLists'][0]['ConstructorStandings']:
        constructor_standings['season'].append(int(json['MRData']['StandingsTable']['StandingsLists'][0]['season']))
        constructor_standings['round'].append(int(json['MRData']['StandingsTable']['StandingsLists'][0]['round']))
        constructor_standings['constructor'].append(item['Constructor']['constructorId'])
        try:
            constructor_standings['constructor_points'].append(int(item['points']))
        except:
            constructor_standings['constructor_points'].append(0)
        constructor_standings['constructor_wins'].append(int(item['wins']))
        constructor_standings['constructor_standings_pos'].append(int(item['position']))
    return  pd.DataFrame(constructor_standings)

def get_race(season, round):
    race = {'season': [],
             'round': [],
             'circuit_id': [],
             'country': [],
             'date': [],
             'url': []}
    url = 'https://ergast.com/api/f1/{}/{}.json'
    try:
        resp = requests.get(url.format(season, round))
    except:
        raise ExternalServiceError()
    item = resp.json()['MRData']['RaceTable']['Races'][0]
    race['season'].append(int(item['season']))
    race['round'].append(int(item['round']))
    race['circuit_id'].append(item['Circuit']['circuitId'])
    race['country'].append(item['Circuit']['Location']['country'])
    race['date'].append(item['date'])
    race['url'].append(item['url'])
    return pd.DataFrame(race)

def get_quali_session(season, round):
    if int(season) > 2024 or int(season) < 2003:
        raise NotFoundError()
    quali = get_quali(season, round)
    driver = get_drivers(season, round)
    constructors = get_constructors(season, round)
    race = get_race(season, round)

    df1 = pd.merge(race, driver, how='left', on=['season', 'round'])
    df2 = pd.merge(df1, quali, how='left', on=['season', 'round', 'driver'])
    final_df = pd.merge(df2, constructors, how='inner', on=['season', 'round', 'constructor'])

    final_df['date'] = pd.to_datetime(final_df.date)
    final_df['date_of_birth'] = pd.to_datetime(final_df.date_of_birth)
    final_df['age'] = final_df.apply(lambda x: relativedelta(x['date'], x['date_of_birth']).years, axis=1)
    final_df.drop(['date', 'date_of_birth'], axis=1, inplace=True)

    for col in ['driver_points', 'driver_wins', 'driver_standings_pos', 'constructor_points',
                'constructor_wins', 'constructor_standings_pos']:
        final_df[col] = final_df[col].fillna(0)

    final_df['qualifying_time'] = final_df.qualifying_time.map(lambda x: 120 if str(x) == '00.000' else (
        float(str(x).split(':')[1]) + (60 * float(str(x).split(':')[0])) if x != 0 else 0))
    final_df = pd.get_dummies(final_df, columns=['circuit_id', 'nationality', 'constructor'])

    for col in headers:
        if col not in final_df.columns:
            final_df[col] = False

    for col in final_df.columns:
        if col not in headers:
            final_df.drop(col, axis=1, inplace=True)
    final_df = final_df[headers]
    for col in booleans:
        final_df[col] = final_df[col].fillna(False)
    final_df.sort_values(['season', 'round', 'grid'], inplace=True)
    final_df.reset_index(drop = True, inplace=True)
    for col in ['driver_points', 'driver_wins', 'driver_standings_pos', 'constructor_points', 'constructor_wins', 'constructor_standings_pos']:
        final_df[col] = final_df[col].map(lambda x: float(x))
    return final_df

