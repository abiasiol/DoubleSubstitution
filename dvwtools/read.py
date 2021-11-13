import pandas as pd
import numpy as np
import re
import os
from tqdm.notebook import tqdm

pd.options.mode.chained_assignment = 'raise'

def read_indices(all_lines):
    """
    Given a list of strings (the content of a DVW file), returns a dictionary with the row indexes
    of key locations in the DVW file
    :param all_lines: list of strings (the content of a DVW file)
    :return: dictionary of row indexes of the different sections of a DVW file
    """
    index_datavolleyscout = [x for x in range(len(all_lines)) if '[3DATAVOLLEYSCOUT]' in all_lines[x]]
    index_match = [x for x in range(len(all_lines)) if '[3MATCH]' in all_lines[x]]
    index_teams = [x for x in range(len(all_lines)) if '[3TEAMS]' in all_lines[x]]
    index_more = [x for x in range(len(all_lines)) if '[3MORE]' in all_lines[x]]
    index_comments = [x for x in range(len(all_lines)) if '[3COMMENTS]' in all_lines[x]]
    index_set = [x for x in range(len(all_lines)) if '[3SET]' in all_lines[x]]
    index_players_int = [x for x in range(len(all_lines)) if '[3PLAYERS-H]' in all_lines[x]]
    index_players_osp = [x for x in range(len(all_lines)) if '[3PLAYERS-V]' in all_lines[x]]
    index_combatt = [x for x in range(len(all_lines)) if '[3ATTACKCOMBINATION]' in all_lines[x]]
    index_setter_call = [x for x in range(len(all_lines)) if '[3SETTERCALL]' in all_lines[x]]
    index_winning_symbols = [x for x in range(len(all_lines)) if '[3WINNINGSYMBOLS]' in all_lines[x]]
    index_reserve = [x for x in range(len(all_lines)) if '[3RESERVE]' in all_lines[x]]
    index_video = [x for x in range(len(all_lines)) if '[3VIDEO]' in all_lines[x]]
    index_scout = [x for x in range(len(all_lines)) if '[3SCOUT]' in all_lines[x]]

    return {'datavolleyscout': index_datavolleyscout,
            'match': index_match,
            'teams': index_teams,
            'more': index_more,
            'comments': index_comments,
            'set': index_set,
            'players_int': index_players_int,
            'players_osp': index_players_osp,
            'combatt': index_combatt,
            'setter_call': index_setter_call,
            'winning_symbols': index_winning_symbols,
            'reserve': index_reserve,
            'video': index_video,
            'scout': index_scout
            }


def read_dvw_file(path, start_from_row):
    """
    Reads the 3SCOUT section of a DVW file and returns a dataframe (with no manipulation)
    :param path: Path of the DVW file
    :param start_from_row: Row index of the [3SCOUT] marker, delimits the begininning of the section to be read
    :return:
    """
    header_columns = ['Rilevato', 'cp', 'idatk', 'vuoto', 'xypartenza', 'controllodir', 'xyarrivo', 'ora',
                      'currentset', 'rotint', 'rotosp', 'numfilm', 'secondipass', 'vuoto2',
                      'p1i', 'p2i', 'p3i', 'p4i', 'p5i', 'p6i',
                      'p1o', 'p2o', 'p3o', 'p4o', 'p5o', 'p6o', 'last']


    dtype_spec = {'xypartenza': str, 'controllodir': str, 'xyarrivo': str, 'ora': str}

    df = pd.read_csv(path, sep=';', skiprows=start_from_row, encoding='iso-8859-1',
                     header=0, names=header_columns, dtype=dtype_spec)

    for i in range(1, 7):
        df[f'p{i}i'] = df[f'p{i}i'].fillna(method="bfill")
        df[f'p{i}o'] = df[f'p{i}o'].fillna(method="bfill")

    df['secondipass'] = df['secondipass'].fillna(method="ffill")
    # df['ora'] = pd.to_datetime(df.ora, format='%H.%M.%S')
    # df = df.drop(columns=['numfilm', 'vuoto2', 'last'])

    return df


def get_team_names(full_file_txt):
    regex_teams = re.compile(
        r'^(\[3TEAMS])(?:\n|\r\n?)'
        r'(?P<TeamInt>\w+)(;)(?P<TeamIntName>[^;]+)(;)(?P<TeamIntFinalSetScore>\d)(;)([^;]*;[^;]*;[^;]*;)(.*)'
        r'(?:\n|\r\n?)'
        r'(?P<TeamOsp>\w+)(;)(?P<TeamOspName>[^;]+)(;)(?P<TeamOspFinalSetScore>\d)(;)([^;]*;[^;]*;[^;]*;)(.*)',
        re.MULTILINE)
    return regex_teams.search(full_file_txt)


def get_match_details(full_file_txt):
    regex_match = re.compile(
        r'^(\[3MATCH])(?:\n|\r\n?)'
        r'(?P<MatchDate>[^;]+)(;)'
        r'(?P<MatchTime>[^;]*)(;)'
        r'(?P<MatchSeason>[^;]*)(;)'
        r'(?P<MatchCompetition>[^;]*)(;)'
        r'(?P<MatchPhase>[^;]*)(;)(.*)',
        re.MULTILINE)
    return regex_match.search(full_file_txt)


def get_players(file, index):
    def drop_players_col(dataf):
        dataf = dataf.drop(
            columns=['id_squadra', 'id_giocatrice', 'libero_flag'])
        return dataf

    header_columns_players = \
        ['id_squadra', 'numero_maglia', 'id_giocatrice', 'primoset', 'secondoset',
         'terzoset', 'quartoset', 'quintoset',
         'codice', 'cognome', 'nome', 'soprannome',
         'libero_flag', 'ruolo', 'straniero']

    last_pos = file.tell()
    df_sq_int = pd.read_csv(file, sep=';', skiprows=index['players_int'][0], encoding='iso-8859-1',
                            header=0, names=header_columns_players,
                            nrows=index['players_osp'][0] - index['players_int'][0] - 1,
                            usecols=range(15))
    file.seek(last_pos)

    df_sq_osp = pd.read_csv(file, sep=';', skiprows=index['players_osp'][0], encoding='iso-8859-1',
                            header=0, names=header_columns_players,
                            nrows=index['combatt'][0] - index['players_osp'][0] - 1,
                            usecols=range(15))
    file.seek(last_pos)

    df_sq_int = assign_position(df_sq_int)
    df_sq_osp = assign_position(df_sq_osp)

    df_sq_int = assign_libero_boolean(df_sq_int)
    df_sq_osp = assign_libero_boolean(df_sq_osp)

    df_sq_int = drop_players_col(df_sq_int)
    df_sq_osp = drop_players_col(df_sq_osp)

    df_sq_int = assign_displayed_player_name(df_sq_int)
    df_sq_osp = assign_displayed_player_name(df_sq_osp)

    return df_sq_int, df_sq_osp


def assign_coordinates(dataf):
    dataf.xypartenza = dataf.xypartenza.replace(r'^\s*$', '-1-1', regex=True)
    dataf.xyarrivo = dataf.xyarrivo.replace(r'^\s*$', '-1-1', regex=True)
    dataf.controllodir = dataf.controllodir.replace(r'^\s*$', '-1-1', regex=True)

    dataf.xypartenza = dataf.xypartenza.fillna('-1-1')
    dataf.xyarrivo = dataf.xyarrivo.fillna('-1-1')
    dataf.controllodir = dataf.controllodir.fillna('-1-1')

    dataf = dataf.assign(x_p=[
        int(xy[-2:])
        if xy
        else -1
        for xy in dataf.xypartenza])
    dataf = dataf.assign(y_p=[
        int(xy[:2])
        if xy
        else -1
        for xy in dataf.xypartenza])

    dataf = dataf.assign(x_dev=[
        int(xy[-2:])
        if xy
        else -1
        for xy in dataf.controllodir])
    dataf = dataf.assign(y_dev=[
        int(xy[:2])
        if xy
        else -1
        for xy in dataf.controllodir])

    dataf = dataf.assign(x_a=[
        int(xy[-2:])
        if xy
        else -1
        for xy in dataf.xyarrivo])
    dataf = dataf.assign(y_a=[
        int(xy[:2])
        if xy
        else -1
        for xy in dataf.xyarrivo])

    dataf= dataf.drop(columns=['xypartenza', 'controllodir', 'xyarrivo'])

    return dataf


def add_scores(dataf):
    """
    Define new columns score_int and score_osp (not team agnostic), and fill them with the score from the Rilevato
    column if Rilevato starts with *p or ap, otherwise it backfills them. Define new columns PuntiInt and PuntiOsp (
    team agnostic) Define columns IsRallyWon if Rilevato starts with *p or ap, otherwise it backfills it. Work with
    REGEX to define named groups (https://docs.python.org/3/howto/regex.html#non-capturing-and-named-groups)

    :param dataf: DVW 3SCOUT dataframe
    :return: DVW 3SCOUT dataframe with added score columns
    """

    regex_score = re.compile(r'[*a]p(?P<score_int>\d\d):(?P<score_osp>\d\d)')
    dataf = dataf.join(dataf.Rilevato.str.extract(regex_score, expand=True))

    dataf[['score_int', 'score_osp']] = dataf[['score_int', 'score_osp']].fillna(method="ffill")
    dataf[['score_int', 'score_osp']] = dataf[['score_int', 'score_osp']].fillna(value='00')

    # Get final score from end of set line (starts with **)
    regex_eos = re.compile(r'\*\*')
    dataf = dataf.assign(final_score_int=[
        score
        if regex_eos.match(rilevato)
        else np.NaN
        for score, rilevato in zip(dataf.score_int, dataf.Rilevato)],
        final_score_osp=[
            score
            if regex_eos.match(rilevato)
            else np.NaN
            for score, rilevato in zip(dataf.score_osp, dataf.Rilevato)],
    )
    dataf[['final_score_int', 'final_score_osp']] = \
        dataf[['final_score_int', 'final_score_osp']].fillna(method="bfill")

    # As the dataframe should be team-agnostic, the final score needs to be expressed based on the team
    # that touches the ball in the specific datarow
    dataf = dataf.assign(FinalSetScore=[
        f'{score_int}-{score_osp}'
        if home_team
        else f'{score_osp}-{score_int}'
        for score_int, score_osp, home_team in
        zip(dataf.final_score_int, dataf.final_score_osp, dataf.is_home_team)],
    )

    def is_rally_won(rilevato, home_team):
        regex_score_int = re.compile(r'[*]p(?P<score_int>\d\d):(?P<score_osp>\d\d)')
        regex_score_osp = re.compile(r'[a]p(?P<score_int>\d\d):(?P<score_osp>\d\d)')
        if home_team:
            if regex_score_int.match(rilevato):
                return True
            else:
                return np.NaN
        else:
            if regex_score_osp.match(rilevato):
                return False
            else:
                return np.NaN

    dataf = dataf.assign(IsRallyWon=[
        is_rally_won(rilevato, home_team)
        for rilevato, home_team in
        zip(dataf.Rilevato, dataf.is_home_team)],
    )
    dataf.IsRallyWon = dataf.IsRallyWon.fillna(method="bfill")
    dataf = dataf.assign(IsRallyWon=[
        not rallyWon
        if not home_team
        else rallyWon
        for rallyWon, home_team in
        zip(dataf.IsRallyWon, dataf.is_home_team)
    ]
    )

    dataf = dataf.assign(PuntiInt=[
        score_int
        if home_team
        else score_osp
        for score_int, score_osp, home_team in
        zip(dataf.score_int, dataf.score_osp, dataf.is_home_team)
    ],
        PuntiOsp=[
            score_osp
            if home_team
            else score_int
            for score_int, score_osp, home_team in
            zip(dataf.score_int, dataf.score_osp, dataf.is_home_team)
        ]
    )

    return dataf


def extract_ball_touch_only(dataf):
    """
    Extracts number, fundamental, type, valutazione, combination, servito, zp, za, za_plus, shot_type,
    block_situation, comments from the column Rilevato, and adds them as columns of the DVW 3SCOUT dataframe :param
    dataf: DVW 3SCOUT dataframe :return: DVW 3SCOUT dataframe with added Rilevato-specific columns
    """
    regex_bt = re.compile(r'[*a](?P<number>\d\d)(?P<fundamental>[SREABDF])(?P<type>[HQTNMO])(?P<valutazione>[#+!\-/=])'
                          r'(?P<combination>[\w~][\w~])?(?P<servito>[\w~])?(?P<zp>[\d~])?(?P<za>[\d~])?(?P<za_plus>['
                          r'\w~])?(?P<shot_type>[\w~])?(?P<block_situation>[\d~])?(?P<comments>[\d~]{1,5})?.?')
    dataf = dataf.join(dataf.Rilevato.str.extract(regex_bt, expand=True))
    dataf = dataf.dropna(subset=['fundamental'])
    return dataf


def add_set_won_column(dataf):
    """
    Add boolean team-agnostic column IsSetWon to DVW dataframe
    :param dataf: DVW 3SCOUT dataframe
    :return: DVW 3SCOUT dataframe with added IsSetWon column
    """
    dataf = dataf.assign(IsSetWon=[
        not (bool(home_team) ^ bool(score_int > score_osp))
        for score_int, score_osp, home_team in
        zip(dataf.final_score_int, dataf.final_score_osp, dataf.is_home_team)],
    )
    return dataf


def assign_match_details(dataf, details):
    dataf = dataf.assign(Competizione=details.group('MatchCompetition'), Round=details.group('MatchPhase'),
                         Date=pd.to_datetime(details.group('MatchDate'), format="%m/%d/%Y"))
    return dataf


def assign_match_name(dataf, details, squadre):
    dataf = dataf.assign(Partita=f'{details.group("MatchDate")} {details.group("MatchCompetition")} '
                                 f'{squadre.group("TeamInt")}-{squadre.group("TeamOsp")} '
                                 f'{squadre.group("TeamIntFinalSetScore")}-{squadre.group("TeamOspFinalSetScore")}')
    return dataf


def assign_libero_boolean(players_list):
    players_list = players_list.assign(libero=[True if libero == 'L' else False for libero in players_list.libero_flag])
    return players_list


def assign_position(players_list):
    condlist = [players_list.ruolo == 1,
                players_list.ruolo == 2,
                players_list.ruolo == 3,
                players_list.ruolo == 4,
                players_list.ruolo == 5]
    choicelist = ['L', 'OH', 'OPP', 'MB', 'S']
    players_list = players_list.assign(position=np.select(condlist, choicelist, default=np.NaN))
    return players_list


def assign_displayed_player_name(dataf):
    dataf = dataf.assign(
        display=[
            soprannome if not soprannome
            else (cognome + ' ' + nome).title()
            for nome, cognome, soprannome in zip(dataf.nome, dataf.cognome, dataf.soprannome)
        ])
    return dataf


def lookup_player_info(df_data, df_data_column, df_players):
    # Match a specific df_data_column to numero_maglia in df_players
    # Example:
    # df_data = df, df_data_column = 'p1i', df_players = df_players_int
    # Given a player number in the column p1i
    # Look in the players list for a corresponding numero_maglia
    # Add the columns codice and display from the players list

    df_data = pd.merge(
        df_data,
        df_players[['numero_maglia', 'codice', 'display']],
        left_on=df_data_column,
        right_on='numero_maglia',
        how='left')
    df_data = df_data.rename(columns={'codice': f'{df_data_column}_code', 'display': f'{df_data_column}_name'})
    df_data = df_data.drop(columns=['numero_maglia'])
    return df_data


def find_court_positions_players_int_info(dataf, players):
    for i in range(1, 7):
        dataf = lookup_player_info(dataf, f'p{i}i', players)
    return dataf


def find_court_positions_players_osp_info(dataf, players):
    for i in range(1, 7):
        dataf = lookup_player_info(dataf, f'p{i}o', players)
    return dataf


def update_player_info(dataf, players_int, players_osp):
    data_int = lookup_player_info(dataf, 'number', players_int)
    data_osp = lookup_player_info(dataf, 'number', players_osp)

    condlist = [dataf.is_home_team == True, dataf.is_home_team == False]

    choicelist = [data_int.number_code, data_osp.number_code]
    dataf = dataf.assign(CodiceGiocatore=np.select(condlist, choicelist, default=np.NaN))

    choicelist = [data_int.number_name, data_osp.number_name]
    dataf = dataf.assign(GiocatoreName=np.select(condlist, choicelist, default=np.NaN))

    return dataf


def add_team_column(dataf, squadre):
    dataf = dataf.assign(SquadraCodice=[
        squadre.group('TeamInt') if home_team is True
        else squadre.group('TeamOsp')
        for home_team in dataf.is_home_team],
        SquadraNome=[
            squadre.group('TeamIntName') if home_team is True
            else squadre.group('TeamOspName')
            for home_team in dataf.is_home_team],
        TeamAvv=[
            squadre.group('TeamOsp') if home_team is True
            else squadre.group('TeamInt')
            for home_team in dataf.is_home_team],
        TeamAvvName=[
            squadre.group('TeamOspName') if home_team is True
            else squadre.group('TeamIntName')
            for home_team in dataf.is_home_team]
    )
    return dataf


def add_match_won(dataf, squadre):
    dataf = dataf.assign(IsMatchWon=[
        not (bool(home_team) ^ bool(
            squadre.group('TeamIntFinalSetScore') > squadre.group('TeamOspFinalSetScore')))
        for home_team in
        dataf.is_home_team]
    )
    return dataf


def get_liberos(players):
    liberos = players[players.libero == True]
    row_count = liberos.shape[0]
    if row_count == 1:
        return {'L1_code': liberos.iloc[0].codice, 'L1_name': liberos.iloc[0].display,
                'L2_code': np.NaN, 'L2_name': np.NaN}
    else:
        return {'L1_code': liberos.iloc[0].codice, 'L1_name': liberos.iloc[0].display,
                'L2_code': liberos.iloc[1].codice, 'L2_name': liberos.iloc[1].display}


def add_sestetto(dataf, players_int, players_osp):
    condlist_int = [
        dataf.rotint == 1,
        dataf.rotint == 6,
        dataf.rotint == 5,
        dataf.rotint == 4,
        dataf.rotint == 3,
        dataf.rotint == 2]
    condlist_osp = [
        dataf.rotosp == 1,
        dataf.rotosp == 6,
        dataf.rotosp == 5,
        dataf.rotosp == 4,
        dataf.rotosp == 3,
        dataf.rotosp == 2]

    def define_player_code_choices(dt, position, code_or_name, team_suffix):
        if position == 'S':
            choicelist = [
                dt[f'p1{team_suffix}_{code_or_name}'],
                dt[f'p6{team_suffix}_{code_or_name}'],
                dt[f'p5{team_suffix}_{code_or_name}'],
                dt[f'p4{team_suffix}_{code_or_name}'],
                dt[f'p3{team_suffix}_{code_or_name}'],
                dt[f'p2{team_suffix}_{code_or_name}'],
            ]
        elif position == 'OH1':
            choicelist = [
                dt[f'p2{team_suffix}_{code_or_name}'],
                dt[f'p1{team_suffix}_{code_or_name}'],
                dt[f'p6{team_suffix}_{code_or_name}'],
                dt[f'p5{team_suffix}_{code_or_name}'],
                dt[f'p4{team_suffix}_{code_or_name}'],
                dt[f'p3{team_suffix}_{code_or_name}'],
            ]
        elif position == 'MB2':
            choicelist = [
                dt[f'p3{team_suffix}_{code_or_name}'],
                dt[f'p2{team_suffix}_{code_or_name}'],
                dt[f'p1{team_suffix}_{code_or_name}'],
                dt[f'p6{team_suffix}_{code_or_name}'],
                dt[f'p5{team_suffix}_{code_or_name}'],
                dt[f'p4{team_suffix}_{code_or_name}'],
            ]
        elif position == 'Opp':
            choicelist = [
                dt[f'p4{team_suffix}_{code_or_name}'],
                dt[f'p3{team_suffix}_{code_or_name}'],
                dt[f'p2{team_suffix}_{code_or_name}'],
                dt[f'p1{team_suffix}_{code_or_name}'],
                dt[f'p6{team_suffix}_{code_or_name}'],
                dt[f'p5{team_suffix}_{code_or_name}'],
            ]
        elif position == 'OH2':
            choicelist = [
                dt[f'p5{team_suffix}_{code_or_name}'],
                dt[f'p4{team_suffix}_{code_or_name}'],
                dt[f'p3{team_suffix}_{code_or_name}'],
                dt[f'p2{team_suffix}_{code_or_name}'],
                dt[f'p1{team_suffix}_{code_or_name}'],
                dt[f'p6{team_suffix}_{code_or_name}'],
            ]
        elif position == 'MB1':
            choicelist = [
                dt[f'p6{team_suffix}_{code_or_name}'],
                dt[f'p5{team_suffix}_{code_or_name}'],
                dt[f'p4{team_suffix}_{code_or_name}'],
                dt[f'p3{team_suffix}_{code_or_name}'],
                dt[f'p2{team_suffix}_{code_or_name}'],
                dt[f'p1{team_suffix}_{code_or_name}'],
            ]
        # elif position == 'L':
        else:
            choicelist = [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]
        return choicelist

    condlist_general = [dataf.is_home_team == True, dataf.is_home_team == False]

    choicelist_general = [np.select(condlist_int, define_player_code_choices(dataf, 'S', 'code', 'i'), default=np.NaN),
                          np.select(condlist_osp, define_player_code_choices(dataf, 'S', 'code', 'o'), default=np.NaN)]
    dataf = dataf.assign(SetCode=np.select(condlist_general, choicelist_general, default=np.NaN))

    choicelist_general = [np.select(condlist_int, define_player_code_choices(dataf, 'S', 'name', 'i'), default=np.NaN),
                          np.select(condlist_osp, define_player_code_choices(dataf, 'S', 'name', 'o'), default=np.NaN)]
    dataf = dataf.assign(SetName=np.select(condlist_general, choicelist_general, default=np.NaN))

    choicelist_general = [
        np.select(condlist_int, define_player_code_choices(dataf, 'OH1', 'code', 'i'), default=np.NaN),
        np.select(condlist_osp, define_player_code_choices(dataf, 'OH1', 'code', 'o'), default=np.NaN)]
    dataf = dataf.assign(OH1Code=np.select(condlist_general, choicelist_general, default=np.NaN))

    choicelist_general = [
        np.select(condlist_int, define_player_code_choices(dataf, 'OH1', 'name', 'i'), default=np.NaN),
        np.select(condlist_osp, define_player_code_choices(dataf, 'OH1', 'name', 'o'), default=np.NaN)]
    dataf = dataf.assign(OH1Name=np.select(condlist_general, choicelist_general, default=np.NaN))

    choicelist_general = [
        np.select(condlist_int, define_player_code_choices(dataf, 'MB2', 'code', 'i'), default=np.NaN),
        np.select(condlist_osp, define_player_code_choices(dataf, 'MB2', 'code', 'o'), default=np.NaN)]
    dataf = dataf.assign(MB2Code=np.select(condlist_general, choicelist_general, default=np.NaN))

    choicelist_general = [
        np.select(condlist_int, define_player_code_choices(dataf, 'MB2', 'name', 'i'), default=np.NaN),
        np.select(condlist_osp, define_player_code_choices(dataf, 'MB2', 'name', 'o'), default=np.NaN)]
    dataf = dataf.assign(MB2Name=np.select(condlist_general, choicelist_general, default=np.NaN))

    choicelist_general = [
        np.select(condlist_int, define_player_code_choices(dataf, 'Opp', 'code', 'i'), default=np.NaN),
        np.select(condlist_osp, define_player_code_choices(dataf, 'Opp', 'code', 'o'), default=np.NaN)]
    dataf = dataf.assign(OppCode=np.select(condlist_general, choicelist_general, default=np.NaN))

    choicelist_general = [
        np.select(condlist_int, define_player_code_choices(dataf, 'Opp', 'name', 'i'), default=np.NaN),
        np.select(condlist_osp, define_player_code_choices(dataf, 'Opp', 'name', 'o'), default=np.NaN)]
    dataf = dataf.assign(OppName=np.select(condlist_general, choicelist_general, default=np.NaN))

    choicelist_general = [
        np.select(condlist_int, define_player_code_choices(dataf, 'OH2', 'code', 'i'), default=np.NaN),
        np.select(condlist_osp, define_player_code_choices(dataf, 'OH2', 'code', 'o'), default=np.NaN)]
    dataf = dataf.assign(OH2Code=np.select(condlist_general, choicelist_general, default=np.NaN))

    choicelist_general = [
        np.select(condlist_int, define_player_code_choices(dataf, 'OH2', 'name', 'i'), default=np.NaN),
        np.select(condlist_osp, define_player_code_choices(dataf, 'OH2', 'name', 'o'), default=np.NaN)]
    dataf = dataf.assign(OH2Name=np.select(condlist_general, choicelist_general, default=np.NaN))

    choicelist_general = [
        np.select(condlist_int, define_player_code_choices(dataf, 'MB1', 'code', 'i'), default=np.NaN),
        np.select(condlist_osp, define_player_code_choices(dataf, 'MB1', 'code', 'o'), default=np.NaN)]
    dataf = dataf.assign(MB1Code=np.select(condlist_general, choicelist_general, default=np.NaN))

    choicelist_general = [
        np.select(condlist_int, define_player_code_choices(dataf, 'MB1', 'name', 'i'), default=np.NaN),
        np.select(condlist_osp, define_player_code_choices(dataf, 'MB1', 'name', 'o'), default=np.NaN)]
    dataf = dataf.assign(MB1Name=np.select(condlist_general, choicelist_general, default=np.NaN))

    liberos_int = get_liberos(players_int)
    liberos_osp = get_liberos(players_osp)

    choicelist_general = [liberos_int['L1_code'], liberos_osp['L1_code']]
    dataf = dataf.assign(L1Code=np.select(condlist_general, choicelist_general, default=np.NaN))

    choicelist_general = [liberos_int['L1_name'], liberos_osp['L1_name']]
    dataf = dataf.assign(L1Name=np.select(condlist_general, choicelist_general, default=np.NaN))

    choicelist_general = [liberos_int['L2_code'], liberos_osp['L2_code']]
    dataf = dataf.assign(L2Code=np.select(condlist_general, choicelist_general, default=np.NaN))

    choicelist_general = [liberos_int['L2_name'], liberos_osp['L2_name']]
    dataf = dataf.assign(L2Name=np.select(condlist_general, choicelist_general, default=np.NaN))

    return dataf


# Add RuoloGiocatore who touches the ball
def add_current_player_position(dataf):
    condlist = [dataf.CodiceGiocatore == dataf['SetCode'],
                dataf.CodiceGiocatore == dataf['OH1Code'],
                dataf.CodiceGiocatore == dataf['MB2Code'],
                dataf.CodiceGiocatore == dataf['OppCode'],
                dataf.CodiceGiocatore == dataf['OH2Code'],
                dataf.CodiceGiocatore == dataf['MB1Code'],
                dataf.CodiceGiocatore == dataf['L1Code'],
                dataf.CodiceGiocatore == dataf['L2Code']]
    choicelist = ['S', 'OH1', 'MB2', 'OPP', 'OH2', 'MB1', 'L', 'L']
    dataf = dataf.assign(CurrentPlayerPosition=np.select(condlist, choicelist, default=np.NaN))
    return dataf


def add_agnostic_rotations(dataf):
    # Manage rotations - make it agnostic
    # if HomeTeam: iz = rotint, az = rotosp
    # if NOT HomeTeam: iz = rotosp, az = rotint

    condlist = [dataf.is_home_team == True, dataf.is_home_team == False]
    choicelist_iz = [dataf.rotint, dataf.rotosp]
    dataf = dataf.assign(iz=np.select(condlist, choicelist_iz, default=0))

    choicelist_az = [dataf.rotosp, dataf.rotint]
    dataf = dataf.assign(az=np.select(condlist, choicelist_az, default=0))

    return dataf


def get_attack_combinations(path, index):
    header_columns_combatt = \
        ['code', 'start', 'zona', 'type', 'description',
         'vuoto', 'coordY', 'coordX', 'servito', 'backrow', 'vuoto2']

    df_att = pd.read_csv(path, sep=';', skiprows=index['combatt'][0], encoding='iso-8859-1',
                         header=0, names=header_columns_combatt,
                         nrows=index['setter_call'][0] - index['combatt'][0] - 1,
                         )
    df_att['SecondaLinea'] = [True if br == 1 else False for br in df_att.backrow]
    df_att = df_att.drop(columns=['vuoto', 'vuoto2', 'backrow'])
    return df_att


def get_setter_calls(path, index):
    header_columns_combalz = \
        ['code', 'vuoto', 'descrizione', 'campo1', 'campo2',
         'campo3', 'campo4', 'campo5', 'campo6', 'campo7', 'vuoto2']

    df_alz = pd.read_csv(path, sep=';', skiprows=index['setter_call'][0], encoding='iso-8859-1',
                         header=0, names=header_columns_combalz,
                         nrows=index['winning_symbols'][0] - index['setter_call'][0] - 1)
    df_alz = df_alz.drop(columns=['vuoto', 'campo1', 'campo2', 'campo3',
                                  'campo4', 'campo5', 'campo6', 'campo7', 'vuoto2'])
    return df_alz


def define_short_game_name(df1):
    # df1['PartitaShortName'] = df1['TeamAvv'] + " (" + df1['Competizione'].apply(lambda x: get_competion_name(x)) + ")"

    df1['PartitaShortName'] = df1['TeamAvv'] + " (" + df1['Competizione'].apply(lambda x: get_competition_name(x)) + ")"
    return df1


def get_competition_acronym(x):
    return "".join(e[0] for e in str(x).split() if not e[0].isdigit())


def get_competition_name(x):
    # return str(x).split(' ', 1)[0]
    return "".join(e for e in str(x).split() if not e[0].isdigit())


def define_set_moment(df1):
    # H = Hard
    # M = Medium
    # E = Easy

    df1['Momento'] = 'Easy'

    part1 = df1['currentset'].values != 5
    part2 = np.logical_or(df1['PuntiInt'] >= 18, df1['PuntiOsp'] >= 18)
    part3 = abs(df1['PuntiInt'].values - df1['PuntiOsp'].values) <= 3
    medium1 = np.logical_and(part1, part2, part3)

    part1 = df1['currentset'].values == 5
    part2 = np.logical_or(df1['PuntiInt'] >= 8, df1['PuntiOsp'] >= 8)
    part3 = abs(df1['PuntiInt'].values - df1['PuntiOsp'].values) <= 3
    medium2 = np.logical_and(part1, part2, part3)

    part1 = df1['currentset'].values != 5
    part2 = np.logical_or(df1['PuntiInt'] >= 21, df1['PuntiOsp'] >= 21)
    part3 = abs(df1['PuntiInt'].values - df1['PuntiOsp'].values) <= 2
    hard1 = np.logical_and(part1, part2, part3)

    part1 = df1['currentset'].values == 5
    part2 = np.logical_or(df1['PuntiInt'] >= 12, df1['PuntiOsp'] >= 12)
    part3 = abs(df1['PuntiInt'].values - df1['PuntiOsp'].values) <= 2
    hard2 = np.logical_and(part1, part2, part3)

    conditions_medium = np.logical_or(medium1, medium2)
    conditions_hard = np.logical_or(hard1, hard2)
    df1.loc[conditions_medium == True, "Momento"] = "Medium"
    df1.loc[conditions_hard == True, "Momento"] = "Hard"

    return df1


valutazioni_cat_type = pd.CategoricalDtype(categories=["#", "+", "!", "-", "/", "="], ordered=True)
fundamental_cat_type = pd.CategoricalDtype(categories=['S', 'R', 'E', 'A', 'B', 'D', 'F'])
momento_cat_type = pd.CategoricalDtype(categories=['Easy', 'Medium', 'Hard'], ordered=True)
set_cat_type = pd.CategoricalDtype(categories=[1, 2, 3, 4, 5], ordered=True)


def proper_data_cast(df_total):
    if not df_total.empty:

        df_total.valutazione = df_total.valutazione.astype(valutazioni_cat_type)
        df_total.cp = df_total.cp.astype(pd.CategoricalDtype(categories=["p", "s"]))
        # p = breakpoint team point, s = sideout team point

        df_total.currentset = df_total.currentset.astype(set_cat_type)
        df_total.idatk = df_total.idatk.astype(pd.CategoricalDtype(categories=['r', 'p', 's']))
        # r = attack after pass, p = breakpoint transition attack, s = sideout transition attack

        df_total.type = df_total.type.astype(pd.CategoricalDtype(categories=['M', 'T', 'H', 'Q', 'O', 'N']))
        df_total.Momento = df_total.Momento.astype(pd.CategoricalDtype(categories=['Easy', 'Medium', 'Hard'], ordered=True))
        df_total.servito = df_total.servito.astype(pd.CategoricalDtype(categories=['F', 'C', 'B', 'P', 'S']))
        df_total.CurrentPlayerPosition = df_total.CurrentPlayerPosition.astype(
            pd.CategoricalDtype(categories=['S', 'OH2', 'OPP', 'OH1', 'MB2', 'L', 'MB1']))
        df_total.fundamental = df_total.fundamental.astype(fundamental_cat_type)
        df_total.CodiceGiocatore = df_total.CodiceGiocatore.astype('category')
        df_total.OH1Code = df_total.OH1Code.astype('category')
        df_total.OH2Code = df_total.OH2Code.astype('category')
        df_total.SetCode = df_total.SetCode.astype('category')
        df_total.OppCode = df_total.OppCode.astype('category')
        df_total.MB1Code = df_total.MB1Code.astype('category')
        df_total.MB2Code = df_total.MB2Code.astype('category')
        df_total.L1Code = df_total.L1Code.astype('category')
        df_total.L2Code = df_total.L2Code.astype('category')
        df_total.PartitaShortName = df_total.PartitaShortName.astype('category')
        df_total.combination = df_total.combination.astype('category')
        df_total.SquadraCodice = df_total.SquadraCodice.astype('category')
        df_total.TeamAvv = df_total.TeamAvv.astype('category')
        df_total.GiocatoreName = df_total.GiocatoreName.astype('category')
        df_total.SquadraNome = df_total.SquadraNome.astype('category')
        df_total.TeamAvvName = df_total.TeamAvvName.astype('category')
        df_total.Partita = df_total.Partita.astype('category')
        df_total.Round = df_total.Round.astype('category')
        df_total.Competizione = df_total.Competizione.astype('category')
        df_total.shot_type = df_total.shot_type.astype('category')
        df_total.zp = df_total.zp.astype('category')
        df_total.za = df_total.za.astype('category')
        df_total.za_plus = df_total.za_plus.astype('category')

        rotazioni_cat_type = pd.CategoricalDtype(categories=[1, 6, 5, 4, 3, 2], ordered=True)
        df_total.iz = df_total.iz.astype(rotazioni_cat_type)
        df_total.az = df_total.az.astype(rotazioni_cat_type)

        for i in range(6):
            df_total[f'p{i + 1}i_code'] = df_total[f'p{i + 1}i_code'].astype('category')
            df_total[f'p{i + 1}o_code'] = df_total[f'p{i + 1}o_code'].astype('category')

        df_total = df_total.astype({'number': 'int8', 'PuntiInt': 'int8', 'PuntiOsp': 'int8',
                                    'block_situation': 'float32',
                                    'x_a': 'int8', 'y_a': 'int8', 'x_p': 'int8',
                                    'y_p': 'int8', 'x_dev': 'int8', 'y_dev': 'int8'})
        df_total.block_situation = df_total.block_situation.astype(
            pd.CategoricalDtype(categories=[1, 2, 3, 4], ordered=True))


    return df_total


def precondition_dvw(dataf, all_dvw_text, percorso_dvw, indices):
    # Drop columns: vuoto, vuoto2, last
    dataf = dataf.drop(columns=['vuoto'])

    # Assign column is_home_team
    # TRUE if Home Team, FALSE if Guest Team
    dataf = dataf.assign(is_home_team=[True if str(x).startswith('*') else False for x in dataf.Rilevato])

    dataf = add_scores(dataf)

    dataf = dataf.drop(columns=['score_int', 'score_osp'])

    dataf = extract_ball_touch_only(dataf)

    dataf = dataf.replace(to_replace='~', value=np.NaN)
    dataf.combination = dataf.combination.replace(to_replace='~~', value=np.NaN)

    dataf = dataf.astype({
        'p1i': 'int8', 'p6i': 'int8', 'p5i': 'int8', 'p4i': 'int8', 'p3i': 'int8', 'p2i': 'int8',
        'p1o': 'int8', 'p6o': 'int8', 'p5o': 'int8', 'p4o': 'int8', 'p3o': 'int8', 'p2o': 'int8',
        'secondipass': 'int32', 'PuntiInt': 'int8', 'PuntiOsp': 'int8',
        'final_score_int': 'int8', 'final_score_osp': 'int8',
        'rotint': 'int8', 'rotosp': 'int8', 'number': 'int16', 'zp': 'category', 'za': 'category',
        'block_situation': 'category', 'comments': 'object'
    })
    # dataf = dataf.astype({
    #     'zp': 'int8', 'za': 'int8', 'block_situation': 'int8'
    # })

    dataf = dataf.assign(secondipass_time=pd.to_timedelta(dataf.secondipass, unit="s"))

    dataf = add_set_won_column(dataf)

    teams = get_team_names(all_dvw_text)
    match_details = get_match_details(all_dvw_text)
    dataf = assign_match_details(dataf, match_details)
    dataf = assign_match_name(dataf, match_details, teams)

    players_int, players_osp = get_players(percorso_dvw, indices)
    dataf = find_court_positions_players_int_info(dataf, players_int)
    dataf = find_court_positions_players_osp_info(dataf, players_osp)
    dataf = update_player_info(dataf, players_int, players_osp)
    dataf = add_team_column(dataf, teams)
    dataf = add_match_won(dataf, teams)

    dataf = add_sestetto(dataf, players_int, players_osp)
    dataf = add_current_player_position(dataf)
    dataf = add_agnostic_rotations(dataf)
    dataf = dataf.drop(columns=['rotint', 'rotosp', 'final_score_int', 'final_score_osp', 'comments'])

    df_combatt = get_attack_combinations(percorso_dvw, indices)
    df_combalz = get_setter_calls(percorso_dvw, indices)

    dataf = dataf.drop(columns=[f'p{j}i' for j in range(1, 7)])
    dataf = dataf.drop(columns=[f'p{j}o' for j in range(1, 7)])

    dataf = define_short_game_name(dataf)
    dataf = define_set_moment(dataf)
    dataf = assign_coordinates(dataf)

    return dataf


def get_attack_df(initial_df):
    data_atk = initial_df
    shifted = data_atk.shift(1)
    data_atk = data_atk.assign(SecondTouchCodiceGiocatore=shifted['CodiceGiocatore'])
    data_atk = data_atk.assign(SecondTouchGiocatoreName=shifted['GiocatoreName'])
    data_atk = data_atk.assign(SecondTouchFundamental=shifted['fundamental'])
    data_atk = data_atk.assign(SecondTouchGrade=shifted['valutazione'])

    shifted = data_atk.shift(2)
    data_atk = data_atk.assign(FirstTouchCodiceGiocatore=shifted['CodiceGiocatore'])
    data_atk = data_atk.assign(FirstTouchGiocatoreName=shifted['GiocatoreName'])
    data_atk = data_atk.assign(FirstTouchFundamental=shifted['fundamental'])
    data_atk = data_atk.assign(FirstTouchGrade=shifted['valutazione'])

    data_atk = data_atk[data_atk.fundamental == 'A']

    second_touch_is_set = data_atk['SecondTouchFundamental'] == 'E'
    data_atk['SecondTouchCodiceGiocatore'] = data_atk['SecondTouchCodiceGiocatore'].where(second_touch_is_set, np.NaN)
    data_atk['SecondTouchGiocatoreName'] = data_atk['SecondTouchGiocatoreName'].where(second_touch_is_set, np.NaN)
    data_atk['SecondTouchGrade'] = data_atk['SecondTouchGrade'].where(second_touch_is_set, np.NaN)
    data_atk['FirstTouchCodiceGiocatore'] = data_atk['FirstTouchCodiceGiocatore'].where(second_touch_is_set, np.NaN)
    data_atk['FirstTouchGiocatoreName'] = data_atk['FirstTouchGiocatoreName'].where(second_touch_is_set, np.NaN)
    data_atk['FirstTouchFundamental'] = data_atk['FirstTouchFundamental'].where(second_touch_is_set, np.NaN)
    data_atk['FirstTouchGrade'] = data_atk['FirstTouchGrade'].where(second_touch_is_set, np.NaN)

    data_atk['SecondTouchGiocatoreName'] = data_atk['SecondTouchGiocatoreName'].cat.add_categories(['No Setter'])
    data_atk['SecondTouchGiocatoreName'].fillna('No Setter', inplace=True)
    data_atk['FirstTouchFundamental'] = data_atk['FirstTouchFundamental'].cat.add_categories(['O'])
    data_atk['FirstTouchFundamental'].fillna('O', inplace=True)
    data_atk = data_atk.dropna(subset=['CurrentPlayerPosition'])
    data_atk = data_atk.assign(Quantity=1)

    data_atk['SecondTouchCodiceGiocatore'] = data_atk['SecondTouchCodiceGiocatore'].astype('category')
    data_atk['SecondTouchGiocatoreName'] = data_atk['SecondTouchGiocatoreName'].astype('category')
    data_atk['SecondTouchGrade'] = data_atk['SecondTouchGrade'].astype(valutazioni_cat_type)
    data_atk['SecondTouchFundamental'] = data_atk['SecondTouchFundamental'].astype(fundamental_cat_type)
    data_atk['FirstTouchCodiceGiocatore'] = data_atk['FirstTouchCodiceGiocatore'].astype('category')
    data_atk['FirstTouchGiocatoreName'] = data_atk['FirstTouchGiocatoreName'].astype('category')
    data_atk['FirstTouchFundamental'] = data_atk['FirstTouchFundamental'].astype(fundamental_cat_type)
    data_atk['FirstTouchGrade'] = data_atk['FirstTouchGrade'].astype(valutazioni_cat_type)

    data_atk['Quantity'] = data_atk['Quantity'].astype('int8')
    return data_atk


def read_dvw_files_in_dir(folder_selected):
    df = pd.DataFrame()

    for file in tqdm(os.listdir(folder_selected)):
        if file.endswith(".dvw"):
            file_path = os.path.join(folder_selected, file)
            with open(file_path) as opened_file:
                print(f'Processing {file_path}', end="...")
                last_pos = opened_file.tell()
                alltext = opened_file.read()
                opened_file.seek(last_pos)
                lines = [line.strip() for line in opened_file]

            indices = read_indices(lines)
            df_aux = read_dvw_file(file_path, indices['scout'][0] + 1)
            df_aux = precondition_dvw(df_aux, alltext, file_path, indices)
            df = df.append(df_aux, ignore_index=True)
            print(f'OK')

    df = proper_data_cast(df)
    return df


def read_rosters_files_in_dir(folder_selected):
    df = pd.DataFrame()

    for file in tqdm(os.listdir(folder_selected)):
        if file.endswith(".sq"):
            file_path = os.path.join(folder_selected, file)
            with open(file_path) as opened_file:
                print(f'Processing {file_path}', end="...")
                df_aux = read_sq_file(file_path)
                df = df.append(df_aux, ignore_index=True)
            print(f'OK')
    return df


def read_sq_file(path):
    header_columns = ['ID', 'code', 'surname', 'birthdate', 'height', 'unk1', 'is_libero', 'unk2', 'name', 'position',
                      'nickname', 'is_excluded', 'is_foreign']

    # dtype_spec = {'height': float}

    df = pd.read_csv(path, sep='\t', skiprows=0, encoding='iso-8859-1',
                     header=0, names=header_columns, index_col=None, usecols=range(len(header_columns)))

    # Get row with title and team name
    title_row = df.iloc[0]

    # remove the header row from the dataset
    df = df[1:]

    df['team_code'] = title_row.loc['ID']
    df['team_name'] = title_row.loc['code']
    df['head_coach'] = title_row.loc['surname']
    df['assistant_coach'] = title_row.loc['birthdate']

    df['birthdate'] = pd.to_datetime(df.birthdate, format='%d/%m/%Y')
    df['is_libero'] = np.where(df.is_libero == 'L', True, False)
    df['height'] = df.height.astype('float')
    df['position'] = df.position.astype('float')
    df = df.drop(columns=['ID', 'unk1', 'unk2', 'is_excluded', 'is_foreign'])

    return df


def get_reception_df(initial_df):
    data_pass = initial_df
    shifted_serve = data_pass.shift(1)
    data_pass = data_pass.assign(ServeCodiceGiocatore=shifted_serve['CodiceGiocatore'])
    data_pass = data_pass.assign(ServeGiocatoreName=shifted_serve['GiocatoreName'])
    data_pass = data_pass.assign(ServeFundamental=shifted_serve['fundamental'])
    data_pass = data_pass.assign(ServeValutazione=shifted_serve['valutazione'])

    shifted_setter = data_pass.shift(-1)
    data_pass = data_pass.assign(SecondTouchCodiceGiocatore=shifted_setter['CodiceGiocatore'])
    data_pass = data_pass.assign(SecondTouchGiocatoreName=shifted_setter['GiocatoreName'])
    data_pass = data_pass.assign(SecondTouchFundamental=shifted_setter['fundamental'])
    data_pass = data_pass.assign(SecondTouchGrade=shifted_setter['valutazione'])
    data_pass = data_pass.assign(SecondTouchCall=shifted_setter['combination'])
    data_pass = data_pass.assign(SecondTouchServito=shifted_setter['servito'])
    data_pass = data_pass.assign(SecondTouchSquadraCodice=shifted_setter['SquadraCodice'])
    data_pass = data_pass.assign(SecondTouchX_start=shifted_setter['x_p'])
    data_pass = data_pass.assign(SecondTouchY_start=shifted_setter['y_p'])

    # data_pass['SecondTouchX_start'] = data_pass['SecondTouchX_start'].where(
    #     (data_pass['SecondTouchX_start'] == - 1) &
    #     (shifted_setter['x_a'] > -1),
    #     shifted_setter['x_a'])



    # data_pass['SecondTouchX_start'] = data_pass['SecondTouchX_start'].where(
    #     (data_pass['SecondTouchX_start'] == - 1) &
    #     (shifted_setter['x_a'] > -1),
    #     shifted_setter['x_a'])
    # data_pass['SecondTouchY_start'] = data_pass['SecondTouchY_start'].where(data_pass['SecondTouchY_start'] == - 1,
    #                                                                         shifted_setter['y_a'])

    shifted_sideout_attack = data_pass.shift(-2)
    data_pass = data_pass.assign(SideoutAttackCodiceGiocatore=shifted_sideout_attack['CodiceGiocatore'])
    data_pass = data_pass.assign(SideoutAttackGiocatoreName=shifted_sideout_attack['GiocatoreName'])
    data_pass = data_pass.assign(SideoutAttackFundamental=shifted_sideout_attack['fundamental'])
    data_pass = data_pass.assign(SideoutAttackGrade=shifted_sideout_attack['valutazione'])
    data_pass = data_pass.assign(SideoutAttackCombination=shifted_sideout_attack['combination'])
    data_pass = data_pass.assign(SideoutAttackSquadraCodice=shifted_sideout_attack['SquadraCodice'])
    data_pass = data_pass.assign(SideoutAttackX_start=shifted_sideout_attack['x_p'])
    data_pass = data_pass.assign(SideoutAttackY_start=shifted_sideout_attack['y_p'])
    data_pass['idatk'] = shifted_sideout_attack['idatk']
    data_pass['cp'] = shifted_sideout_attack['cp']

    data_pass = data_pass.loc[data_pass.fundamental == 'R']

    # Check where the second touch is a SET and from the same team
    second_touch_is_set = \
        (data_pass['SecondTouchFundamental'] == 'E') & \
        (data_pass['SquadraCodice'] == data_pass['SecondTouchSquadraCodice'])

    data_pass['SecondTouchCodiceGiocatore'] = data_pass['SecondTouchCodiceGiocatore'].where(second_touch_is_set, np.NaN)
    data_pass['SecondTouchGiocatoreName'] = data_pass['SecondTouchGiocatoreName'].where(second_touch_is_set, np.NaN)
    data_pass['SecondTouchGrade'] = data_pass['SecondTouchGrade'].where(second_touch_is_set, np.NaN)
    data_pass['SecondTouchCall'] = data_pass['SecondTouchCall'].where(second_touch_is_set, np.NaN)
    data_pass['SecondTouchServito'] = data_pass['SecondTouchServito'].where(second_touch_is_set, np.NaN)
    data_pass['SecondTouchFundamental'] = data_pass['SecondTouchFundamental'].where(second_touch_is_set, np.NaN)

    data_pass['SecondTouchX_start'] = data_pass['SecondTouchX_start'].where(second_touch_is_set, np.NaN)
    data_pass['SecondTouchY_start'] = data_pass['SecondTouchY_start'].where(second_touch_is_set, np.NaN)


    data_pass['SideoutAttackCodiceGiocatore'] = data_pass['SideoutAttackCodiceGiocatore'].where(second_touch_is_set, np.NaN)
    data_pass['SideoutAttackGiocatoreName'] = data_pass['SideoutAttackGiocatoreName'].where(second_touch_is_set, np.NaN)
    data_pass['SideoutAttackFundamental'] = data_pass['SideoutAttackFundamental'].where(second_touch_is_set, np.NaN)
    data_pass['SideoutAttackGrade'] = data_pass['SideoutAttackGrade'].where(second_touch_is_set, np.NaN)
    data_pass['SideoutAttackCombination'] = data_pass['SideoutAttackCombination'].where(second_touch_is_set, np.NaN)
    data_pass['SecondTouchSquadraCodice'] = data_pass['SecondTouchSquadraCodice'].where(second_touch_is_set, np.NaN)

    data_pass['SideoutAttackX_start'] = data_pass['SideoutAttackX_start'].where(second_touch_is_set, np.NaN)
    data_pass['SideoutAttackY_start'] = data_pass['SideoutAttackY_start'].where(second_touch_is_set, np.NaN)



    # Check where the third touch is ATTACK and from the same team
    third_touch_is_attack = (data_pass['SideoutAttackFundamental'] == 'A') & \
                            (data_pass['SquadraCodice'] == data_pass['SideoutAttackSquadraCodice'])

    data_pass['SideoutAttackCodiceGiocatore'] = data_pass['SideoutAttackCodiceGiocatore'].where(third_touch_is_attack,
                                                                                                np.NaN)
    data_pass['SideoutAttackGiocatoreName'] = data_pass['SideoutAttackGiocatoreName'].where(third_touch_is_attack, np.NaN)
    data_pass['SideoutAttackFundamental'] = data_pass['SideoutAttackFundamental'].where(third_touch_is_attack, np.NaN)
    data_pass['SideoutAttackGrade'] = data_pass['SideoutAttackGrade'].where(third_touch_is_attack, np.NaN)
    data_pass['SideoutAttackCombination'] = data_pass['SideoutAttackCombination'].where(third_touch_is_attack, np.NaN)
    data_pass['SideoutAttackSquadraCodice'] = data_pass['SideoutAttackSquadraCodice'].where(third_touch_is_attack,
                                                                                            np.NaN)
    data_pass['cp'] = data_pass['cp'].where(third_touch_is_attack, np.NaN)
    data_pass['idatk'] = data_pass['idatk'].where(third_touch_is_attack, np.NaN)

    data_pass['SecondTouchGiocatoreName'] = data_pass['SecondTouchGiocatoreName'].cat.add_categories(['No Setter'])
    data_pass['SecondTouchGiocatoreName'].fillna('No Setter', inplace=True)
    new_cat_set_valutazione = pd.CategoricalDtype(categories=["#", "+", "!", "-", "/", "=", 'No Set'],
                                                     ordered=True)
    data_pass['SecondTouchGrade'] = data_pass['SecondTouchGrade'].astype(new_cat_set_valutazione)
    data_pass['SecondTouchGrade'].fillna('No Set', inplace=True)
    data_pass['SideoutAttackGiocatoreName'] = data_pass['SideoutAttackGiocatoreName'].cat.add_categories(['No Attacker'])
    data_pass['SideoutAttackGiocatoreName'].fillna('No Attacker', inplace=True)
    new_cat_attack_valutazione = pd.CategoricalDtype(categories=["#", "+", "!", "-", "/", "=", 'No Attack'], ordered=True)
    data_pass['SideoutAttackGrade'] = data_pass['SideoutAttackGrade'].astype(new_cat_attack_valutazione)
    data_pass['SideoutAttackGrade'].fillna('No Attack', inplace=True)

    data_pass = data_pass.dropna(subset=['CurrentPlayerPosition'])
    data_pass = data_pass.assign(Quantity=1)

    data_pass['ServeCodiceGiocatore'] = data_pass['ServeCodiceGiocatore'].astype('category')
    data_pass['ServeGiocatoreName'] = data_pass['ServeGiocatoreName'].astype('category')
    data_pass['ServeValutazione'] = data_pass['ServeValutazione'].astype(valutazioni_cat_type)
    data_pass['ServeFundamental'] = data_pass['ServeFundamental'].astype('category')

    data_pass['SecondTouchCodiceGiocatore'] = data_pass['SecondTouchCodiceGiocatore'].astype('category')
    data_pass['SecondTouchGiocatoreName'] = data_pass['SecondTouchGiocatoreName'].astype('category')
    data_pass['SecondTouchFundamental'] = data_pass['SecondTouchFundamental'].astype('category')
    data_pass['SecondTouchCall'] = data_pass['SecondTouchCall'].astype('category')
    data_pass['SecondTouchServito'] = data_pass['SecondTouchServito'].astype('category')
    data_pass['SecondTouchSquadraCodice'] = data_pass['SecondTouchSquadraCodice'].astype('category')

    data_pass['SideoutAttackCodiceGiocatore'] = data_pass['SideoutAttackCodiceGiocatore'].astype('category')
    data_pass['SideoutAttackGiocatoreName'] = data_pass['SideoutAttackGiocatoreName'].astype('category')
    data_pass['SideoutAttackGrade'] = data_pass['SideoutAttackGrade'].astype(new_cat_attack_valutazione)
    data_pass['SideoutAttackFundamental'] = data_pass['SideoutAttackFundamental'].astype('category')
    data_pass['SideoutAttackCombination'] = data_pass['SideoutAttackCombination'].astype('category')
    data_pass['currentset'] = data_pass['currentset'].astype(set_cat_type)

    data_pass['Quantity'] = data_pass['Quantity'].astype('int8')
    data_pass = data_pass.drop(columns=['combination', 'servito', 'block_situation'])
    return data_pass
