import io

import pandas as pd

import dvwtools.read as dv
import numpy as np

pd.set_option('max_columns', None)

from enum import Enum


class Team(Enum):
    HOME = '*'
    GUEST = 'a'


# Function to read the main body of the scout file
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

    last_pos = path.tell()
    df = pd.read_csv(path, sep=';', skiprows=start_from_row, encoding='iso-8859-1',
                     header=0, names=header_columns, dtype=dtype_spec)

    for i in range(1, 7):
        df[f'p{i}i'] = df[f'p{i}i'].fillna(method="bfill")
        df[f'p{i}o'] = df[f'p{i}o'].fillna(method="bfill")

    df['secondipass'] = df['secondipass'].fillna(method="ffill")

    df = df.astype({
        'p1i': 'int8', 'p6i': 'int8', 'p5i': 'int8', 'p4i': 'int8', 'p3i': 'int8', 'p2i': 'int8',
        'p1o': 'int8', 'p6o': 'int8', 'p5o': 'int8', 'p4o': 'int8', 'p3o': 'int8', 'p2o': 'int8',
        'secondipass': 'int32', 'rotint': 'int8', 'rotosp': 'int8', 'cp': 'str', 'idatk': 'str',
        'vuoto': 'str', 'xypartenza': 'str', 'controllodir': 'str', 'xyarrivo': 'str', 'ora': 'str',
        'currentset': 'str'
    })

    df.rotint = df.rotint.replace(to_replace=0, value=np.NaN)
    df.rotosp = df.rotosp.replace(to_replace=0, value=np.NaN)

    df.rotint = df.rotint.fillna(method='ffill')
    df.rotosp = df.rotosp.fillna(method='ffill')
    df.rotint = df.rotint.astype(int)
    df.rotosp = df.rotosp.astype(int)
    # df['ora'] = pd.to_datetime(df.ora, format='%H.%M.%S')
    # df = df.drop(columns=['numfilm', 'vuoto2', 'last'])

    path.seek(last_pos)
    return df


# Find if there's a setter code in the rally associated to the old setter (e.g., *04EH# or similar)
# Assign it to the new setter

def change_setter_codes(data, team: Team, old_setter, new_setter, current_rally_index, log_string):
    current_rally_rows = data.rally_index == current_rally_index

    pattern_escaped = '\*'
    if team == Team.GUEST:
        pattern_escaped = 'a'

    pattern = f'({pattern_escaped})({old_setter:02})(E)([HMTQNOU])([#\+\-\!/=])'
    filter_set_codes = data.Rilevato.str.match(pattern)

    set_codes_old = data.loc[(current_rally_rows) & (filter_set_codes)]

    if len(set_codes_old) > 0:

        # display(df_copy.loc[filtered, ["Rilevato"]].replace(to_replace=f'{old_setter:02}', value=f'{new_setter:02}', regex=True))
        data.loc[(current_rally_rows) & (filter_set_codes), 'Rilevato'] = data.loc[
            (current_rally_rows) & (filter_set_codes), 'Rilevato'].replace(to_replace=f'{old_setter:02}',
                                                                           value=f'{new_setter:02}', regex=True)
        # found = df_copy.loc[(df_copy.Rilevato.str.match(pattern)) & (df_copy.rally_index == j)]

        pattern_escaped = '\*'
        if team == Team.GUEST:
            pattern_escaped = 'a'

        pattern = f'({pattern_escaped})({new_setter:02})(E)([HMTQNOU])([#\+\-\!/=])'
        filter_set_codes = data.Rilevato.str.match(pattern)
        set_codes_new = data.loc[(current_rally_rows) & (filter_set_codes)]

        for timecode, rilevato, rilevato_new in zip(set_codes_old.timecode, set_codes_old.Rilevato,
                                                    set_codes_new.Rilevato):
            log_string += (f'\n{str(timecode)[-8:]};'
                  f'{rilevato};'
                  f'{current_rally_index};'
                  f'Changed setter number to {rilevato_new}; {team.name.lower()}')

    return data, log_string


def change_setter_assignment_code(data, team: Team, old_setter, new_setter, current_rally_index, log_string):
    current_rally_rows = data.rally_index == current_rally_index
    change_setter_filter = data.Rilevato.str.startswith(f'{team.value}P')
    change_setter_rows = data.loc[(current_rally_rows) & (change_setter_filter)]
    is_change_setter_present = len(change_setter_rows)

    if is_change_setter_present:
        # The setter change is present in the rally, fix the old setter with the new setter
        index_setter_change = change_setter_rows.index.item()
        # print(index_setter_change)

        verify_old_setter = data.loc[index_setter_change, "Rilevato"]
        data.loc[index_setter_change, "Rilevato"] = f'{team.value}P{new_setter:.0f}'
        verify_new_setter = data.loc[index_setter_change, "Rilevato"]

        log_string += (f'\n{str(data.loc[index_setter_change, "timecode"])[-8:]};'
              f'{data.loc[index_setter_change, "Rilevato"]};'
              f'{current_rally_index:.0f};'
              f' Changed setter assignment code from {verify_old_setter} to {verify_new_setter}; {team.name.lower()}')
    return data, log_string


def change_rotation_in_rally(data, team: Team, old_rotation, new_rotation, old_setter, new_setter, players_list,
                             current_rally_index, serve_line, log_string):
    # Find if there's a rotation change code in the rally (e.g., *z4 or az1 or similar)
    current_rally_rows = data.rally_index == current_rally_index
    changerot_filter = data.Rilevato.str.startswith(f'{team.value}z{old_rotation}')
    changeof_rotation_rows = data.loc[(current_rally_rows) & (changerot_filter)]
    is_changerot_present = len(changeof_rotation_rows)

    if is_changerot_present:

        old_setter_info_display = players_list[["cognome", "nome", "soprannome"]].loc[
            players_list.numero_maglia == old_setter].to_string(formatters=formatters_nome, header=False, index=False)
        new_setter_info_display = players_list[["cognome", "nome", "soprannome"]].loc[
            players_list.numero_maglia == new_setter].to_string(formatters=formatters_nome, header=False, index=False)

        # The rotation change is present in the rally, fix the old_rotation with the new_rotation
        index_rotation_change = changeof_rotation_rows.index.item()
        # print(index_rotation_change)

        # Change the rotation row to the new rotation in the code
        verify_old_row = data.loc[index_rotation_change, "Rilevato"]
        data.loc[index_rotation_change, "Rilevato"] = f'{team.value}z{new_rotation}'
        verify_new_row = data.loc[index_rotation_change, "Rilevato"]

        # Change the rotation rotint or rotosp in all the rally
        rot_suffix = 'int'
        if team == Team.GUEST:
            rot_suffix = 'osp'

        # Find next change of rotation or end of set reference
        pattern_escaped = '\*'
        if team == Team.GUEST:
            pattern_escaped = 'a'

        pattern_point = f'({pattern_escaped}z\d)|(\*\*\dset)'
        found_pattern_rows = data.Rilevato.str.match(pattern_point)
        next_rally_rows = data.rally_index == current_rally_index + 1

        rotation_change_next = data.loc[(next_rally_rows) & (found_pattern_rows)]
        index_rotation_change_next = rotation_change_next.index.item()
        data.loc[(next_rally_rows) & (data.index < index_rotation_change_next), f"rot{rot_suffix}"] = int(new_rotation)

        data.loc[(current_rally_rows) & (data.index >= index_rotation_change), f"rot{rot_suffix}"] = int(new_rotation)

        log_string += (f'\n{str(serve_line.timecode)[-8:]}; {str(serve_line.Rilevato)}; {current_rally_index:.0f};'
              f' {str(team.name).title()} rotation: {old_rotation}, Setter: {old_setter:.0f} {old_setter_info_display}; {team.name.lower()}')

        # print(f'old {verify_old_row} new {verify_new_row}')

        log_string += (f'\n{str(serve_line.timecode)[-8:]}; {str(serve_line.Rilevato)}; {current_rally_index:.0f};'
              f' Change {str(team.name).lower()} rotation to: {new_rotation}, Setter: {new_setter:.0f} {new_setter_info_display}; {team.name.lower()}')

        # Find if there's a setter change code in the rally (e.g., *P1 or aP13 or similar)
        data, log_string = change_setter_assignment_code(data, team, old_setter, new_setter, current_rally_index, log_string)

        # Find if there's a setter code in the rally associated to the old setter (e.g., *04EH# or similar)
        # Assign it to the new setter
        data, log_string = change_setter_codes(data, team, old_setter, new_setter, current_rally_index, log_string)



    return data, log_string



def fix_double_sub(data, team: Team, players, log_string):
    suffix_team = 'home'
    suffix_rot = 'int'
    suffix_court_pos = 'i'

    if team == Team.GUEST:
        suffix_team = 'guest'
        suffix_rot = 'osp'
        suffix_court_pos = 'o'

    # print(f'S {suffix_team} {suffix_rot}')
    # Flag rows with discrepancy in setter
    rows_discrepancies = data[f'setter_loc_{suffix_team}'] != data[f'rot{suffix_rot}']

    # Get rally index where discrepancies are
    rally_discrepancies = data.loc[rows_discrepancies, 'rally_index'].unique()
    # print(rally_discrepancies)

    serves = data.Rilevato.str.match('([\*a])(\d\d)(S)([HMTQNOU])([#\+\-\!/=])')
    end_of_set = data.Rilevato.str.match('(\*\*)')

    for j in rally_discrepancies:
        current_rally_rows = data.rally_index == j

        # Get the first value of setter_loc_home in the rally
        setter_location = data.loc[current_rally_rows, f'setter_loc_{suffix_team}'].iloc[0]

        if setter_location == 0:
            # No setter has been detected in the current rows
            # Get the serve line in the current rally

            serve_line = data.loc[(current_rally_rows) & (serves)]

            if len(serve_line) == 1:
                serve_line = serve_line.iloc[0]
                log_string += (f'\n{str(serve_line.timecode)[-8:]}; {str(serve_line.Rilevato)}; {j:.0f}; No setter found; {team.name.lower()}')

        else:
            # Verify that the discrepancy does not correspond to an end of set
            end_of_set_rows = data.loc[(current_rally_rows) & (end_of_set)]

            if len(end_of_set_rows) == 0:

                # Get the serve in the rally
                serve_line = data.loc[(current_rally_rows) & (serves)]

                if len(serve_line) == 1:
                    serve_line = serve_line.iloc[0]
                    new_setter_pos = serve_line[f'setter_loc_{suffix_team}']
                    new_setter = serve_line[f'p{new_setter_pos}{suffix_court_pos}']
                    old_rotation = serve_line[f'rot{suffix_rot}']
                    old_setter = serve_line[f'p{old_rotation}{suffix_court_pos}']
                    # print(f'NS {new_setter}; OS {old_setter}; NR {new_setter_pos}; OR {old_rotation}')

                    if (new_setter != old_setter) | (new_setter_pos != old_rotation):
                        data, log_string = change_rotation_in_rally(data, team, old_rotation, new_setter_pos, old_setter,
                                                        new_setter, players, j, serve_line, log_string)
    return data, log_string





# Get setters
def get_setters(players_list):
    return players_list[players_list.ruolo == 5]


formatters_nome = {'numero_maglia': '#{:d}'.format, 'soprannome': '({})'.format}


def setters_list(df, team: Team, log_string):
    for numero, cognome, nome, soprannome in zip(df.numero_maglia, df.cognome, df.nome, df.soprannome):
        log_string += f'00:00:00;;0;#{numero:d} {cognome.strip().title()} {nome.strip().title()} ({soprannome.strip().title()}); {team.name.lower()}\n'
    return log_string


def perform_double_substitution(file):


    # if file_path.endswith(".dvw"):
    #     with open(file_path) as opened_file:
    #         print(f'Processing {file_path}', end="...")

    last_pos = file.tell()
    lines = [line.strip() for line in file]
    file.seek(last_pos)

    indices = dv.read_indices(lines)

    # Read dvw file and get teams list
    df_aux = read_dvw_file(file, indices['scout'][0])
    players_int, players_osp = dv.get_players(file, indices)


    df_copy = df_aux.copy()
    df_copy = df_copy.assign(timecode=pd.to_timedelta(df_copy.secondipass, unit='s'))

    setters_int = get_setters(players_int)
    setters_osp = get_setters(players_osp)
    
    log_string = ''

    log_string += ("00:00:00;;0;Home Setters;home\n")
    log_string = setters_list(setters_int, Team.HOME, log_string)

    log_string += ("\n00:00:00;;0;Guest Setters;guest\n")
    log_string = setters_list(setters_osp, Team.GUEST, log_string)

    # Count rallies by assigned points
    pattern_point = '([\*a]p\d\d:\d\d)|(\*\*\dset)'
    rallies_count = df_copy.Rilevato.str.count(pattern_point).sum()
    log_string += (f'\n00:00:00;;0;Number of rallies: {rallies_count};\n')

    # Assign number of rally as attribute to each row
    rally_index = np.arange(0, rallies_count)
    df_copy['rally_index'] = np.NaN
    df_copy.loc[df_copy.Rilevato.str.match(pattern_point), 'rally_index'] = rally_index
    df_copy.rally_index = df_copy.rally_index.fillna(method='bfill')

    # For each row,find in which position the setter is (if any)
    df_copy = df_copy.assign(setter_loc_home=np.NaN)
    df_copy = df_copy.assign(setter_loc_guest=np.NaN)

    for i in range(1, 7):
        filter_rows = df_copy[f'p{i}i'].isin(setters_int.numero_maglia)
        df_copy.loc[filter_rows, 'setter_loc_home'] = i

        filter_rows = df_copy[f'p{i}o'].isin(setters_osp.numero_maglia)
        df_copy.loc[filter_rows, 'setter_loc_guest'] = i

    # If no setter is found, flag with a 0
    df_copy.setter_loc_home = df_copy.setter_loc_home.fillna(0)
    df_copy.setter_loc_home = df_copy.setter_loc_home.astype(int)

    df_copy.setter_loc_guest = df_copy.setter_loc_guest.fillna(0)
    df_copy.setter_loc_guest = df_copy.setter_loc_guest.astype(int)

    df_copy, log_string = fix_double_sub(df_copy, Team.HOME, players_int, log_string)
    df_copy, log_string = fix_double_sub(df_copy, Team.GUEST, players_osp, log_string)

    df_copy = df_copy.drop(columns={'rally_index', 'setter_loc_home', 'setter_loc_guest', 'timecode'})

    df_copy = df_copy.replace('nan', '')
    df_copy = df_copy.replace('NaT', '')
    df_copy = df_copy.fillna('')

    file_out = io.StringIO()
    file_out.write('\n'.join(lines[:indices['scout'][0] + 1]))
    file_out.write('\n')
    df_copy.to_csv(file_out, header=None, index=None, sep=';', mode='a')

    return log_string, file_out


