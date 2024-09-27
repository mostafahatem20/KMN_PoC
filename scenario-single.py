#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import re
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.pyplot import cycler # plot styling
import argparse
import math
import json
import datetime
import matplotlib.colors
import colorsys
from typing import Dict, List, Tuple, Callable, Any, Iterable, TypeVar, Generic
from colorama import init, Fore, Style
import shutil
import sys, os
from typing import Dict

init()
np.set_printoptions(linewidth=shutil.get_terminal_size((80, 20)).columns) # auto-detect terminal width (with fallback)
sys.path.append(os.path.dirname(os.path.realpath(__file__))) # this works regardless of the cwd


def mergeDicts(a : Dict, b : Dict) -> Dict:
    temp = a.copy()
    temp.update(b)
    return temp

def debug():
    """!Only for debugging!: Open interactive python interpreter with the caller's environment"""
    import code
    import inspect
    if "pandas" in sys.modules: 
        # to make sure name "pandas" is present
        import pandas
        pandas.set_option('display.width', None) # auto-detect terminal width
    if "numpy" in sys.modules:
        import numpy
        import shutil
        numpy.set_printoptions(linewidth=shutil.get_terminal_size((80, 20)).columns)
    
    code.interact(local=mergeDicts(inspect.currentframe().f_back.f_globals, inspect.currentframe().f_back.f_locals))

def print_warn(msg):
    if "colorama" in sys.modules:
        from colorama import Fore
        print(f"{Fore.MAGENTA}{msg}{Fore.RESET}", file=sys.stderr)
    else:
        print(msg, file=sys.stderr)

_darkYellow = "#fab20b" # TUM dark yellow
_orange = "#ff7f0e" # the less beautiful TUM orange: #E37222
_red = "#d62728"
_green = "#A2AD00"
_green = "#A2AD00"
_darkGreen= "#008940" #TUMDarkGreen
_lighterBlue = "#5aa3d8"
_pantone300 = "#3070b3" # TUM blue
_turquois = "#017a8c" # TUMTurquois
# non-TUM:
_purple = "#75005f"
_bluepurple = "#681edf"

colors = [_green, _pantone300, _orange, _darkGreen, _lighterBlue, _darkYellow, _red, _purple, _turquois, _bluepurple]
colors_five_two_one = [_darkYellow,_orange, _red, _purple, _bluepurple, _lighterBlue, _pantone300, _darkGreen]
#def transform_for_alpha(c: str):
#    r,b,g = matplotlib.colors.to_rgb(c)
#    h,s,v = colorsys.rgb_to_hsv(r,b,g)
#    s = min(1, s + s*0.6) # increase saturation so that color is nicer with some alpha on white background
#    return colorsys.hsv_to_rgb(h,s,v)
#""" colors suitable to be used with some alpha """
#colors_alpha = list(map(transform_for_alpha, colors))
##plt.setp(handle, color='g')
# c2.concat(c) cycler("color", [])
color_cycles = { # maps number of needed styles to the according color schemes
    2: [_green, _pantone300],
    3: [_green, _pantone300, _orange],
    4: [_green, _pantone300, _red, _orange],
    5: [_green, _darkGreen, _pantone300, _darkYellow, _red],
    6: [_green, _darkGreen, _lighterBlue, _pantone300, _darkYellow, _orange],
    7: [_darkYellow, _orange, _red, _green, _darkGreen, _lighterBlue, _pantone300],
    8: [_green, _darkGreen, _lighterBlue, _pantone300, _red, _darkYellow, _orange, _purple],
    10: colors,
}
def color_handles(handles: Iterable):
    """ applies an appropriate color cycle/scheme to the given mathplotlib artist handles """
    handles = list(handles)
    colors = color_cycles.get(len(handles), None)
    if colors is None:
        print_warn(f"No color cycle for {len(handles)}!")
    else:
        for (handle, color) in zip(handles, colors):
            plt.setp(handle, color=color)

def color_violinplot_handle(handle, color):
    for pc in handle["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(0.4) # 0.2 seemed to be a little too weak in print
    for item, handle in handle.items(): # set color for all components, e.g. "cmaxes", "cmins", "cbars"
        if item != "bodies":
            handle.set_color(color)

def nsToUs(x):
    return x / math.pow(10, 3)

def nsToMs(x):
    return x / math.pow(10, 6)

def findCsvSubdirHeuristic(path: str):
    dirpaths = []
    for dirpath, dirs, files in os.walk(path, followlinks=True):
        for filename in files:
            if re.match("n.*.csv", filename) is not None:
                dirpaths.append(dirpath)
    return dirpaths

def parse_json_files(directory: str) -> List[Dict]:
    """
    Parse JSON files from all 'node' subdirectories in the given directory.

    Args:
    - directory: The top-level directory containing experiment folders.

    Returns:
    - A list of parsed JSON data from all files found within 'node*' subdirectories.
    """
    json_data = []

    # Traverse the experiment directory
    for experiment_folder in os.listdir(directory):
        experiment_path = os.path.join(directory, experiment_folder)

        # Check if it's a directory
        if os.path.isdir(experiment_path):
            # Look for directories starting with 'node'
            for subdir in os.listdir(experiment_path):
                if subdir.startswith('node'):
                    node_path = os.path.join(experiment_path, subdir)

                    # Check inside the node directory for JSON files
                    if os.path.isdir(node_path):
                        for json_file in os.listdir(node_path):
                            if json_file.endswith(".json"):
                                json_file_path = os.path.join(node_path, json_file)
                                with open(json_file_path, 'r') as f:
                                    json_content = json.load(f)
                                    json_data.append(json_content)
    
    return json_data

def findFrostDalekBenchmarkResultJsonHeuristic(path: str):
    for dirpath, dirs, files in os.walk(path, followlinks=True):
        for filename in files:
            if filename == "frost_dalek_benchmarks.json":
                return os.path.join(dirpath, filename)
    return None

def findConfigJsonHeuristic(path: str):
    """ finds the path to allocation.json containing metadata such as the date"""
    pos_alloc_file = ["config", "allocation.json"]
    temp = os.path.join(path, *pos_alloc_file)
    if os.path.exists(temp):
        return temp

    temp = os.path.join(path, "..", *pos_alloc_file)
    if os.path.exists(temp):
        return temp

    return None

def parseTestcaseDesc(desc: str) -> (str, Dict):
    """ The result csv files contain the test case and its parameters, which we parse here
    returns: (testcase_name, argdict) """
    print(desc)
    if "dkg" in desc.split("_"):
        desc = desc.split("_")[1]
    elif "sign" in desc.split("_"):
        desc = desc.split("_")[1] + "_" + desc.split("_")[2]
    else:
        desc = desc.split("_")[1] + "_" + desc.split("_")[2]
    if desc == "dkg":
        return ("dkg", {})
    
    matches = re.search('^prep(rocess)?_nn(?P<%s>[0-9]+)$' % "num_noncepairsGroup", desc)
    if matches is not None:
        num_noncepairs = int(matches.group("num_noncepairsGroup"))
        #return lambda x: plot_preprocess(num_noncepairs, x)
        return ("preprocess", {"num_noncepairs": num_noncepairs})
    
    matches = re.search('^sign?_l(?P<%s>[0-9]+)$' % "msg_lenGroup", desc)
    print(matches)
    if matches is not None:
        msg_len = int(matches.group("msg_lenGroup"))
        return ("sign", {"msg_len": msg_len})
    
    print(f"Unknown testcase description '{desc}'", file=sys.stderr)
    assert(False)

def get_range_params(values: List) -> str: 
    """ return concise description for start;end;stepsize in given array or range (suitable as filename) """
    if len(values) == 1:
        return str(values[0])
    values = sorted(values)
    start = values[0]
    stop = values[-1]
    stepwidth = values[1] - values[0]
    reconstructed = range(start, stop+1, stepwidth)
    if np.array_equal(reconstructed, values):
        if stepwidth == 1:
            return f"{start}-{stop}"
        else:
            return f"{start}-{stop}s{stepwidth}"
    else:
        return f"{','.join(map(lambda x: str(x), values))}"

def sample_in_list(len_sample_set: int, num_samples: int): # -> list-like
    """ samples with equal distance and returns indices """
    if num_samples == 1:
        return [math.floor(len_sample_set/2)] # use value in the middle if only one is requested
    def to_int(x):
        if x < len_sample_set/2:
            return math.floor(x)
        else:
            return math.ceil(x)
    num_samples = min(num_samples, len_sample_set)
    return map(to_int, np.linspace(0, len_sample_set-1, num=num_samples))

def sample_t_values(values: List, num_samples: int) -> Iterable:
    """ samples with equal distance and returns values. Ensures that t=2 is returned when within/near range """
    def to_int(x):
        if x < len(values)/2:
            return math.floor(x)
        else:
            return math.ceil(x)

    if num_samples == 1:
        indices = [math.floor(len(values)/2)] # use value in the middle if only one is requested
    else:
        num_samples = min(num_samples, len(values))
        indices = list(map(to_int, np.linspace(0, len(values)-1, num=num_samples)))

    # nudge towards t=2
    if 2 in values and 2 not in map(lambda i: values[i], indices):
        for index, i in enumerate(indices):
            if i > 0 and values[i-1] == 2:
                indices = indices[:i-1] + [i-1] + indices[i:]
                break
            if values[i] == 1:
                assert(values[i+1] == 2)
                indices = indices[:i] + [i+1] + indices[i+1:]
                break
    return map(lambda i: values[i], indices)

def filterlist(bottom, top, iterable):
    """ removes all list elements not within the given bounds """
    below = lambda y: y < top
    above = lambda y: y > bottom
    return list(filter(above, filter(below, iterable)))

def violin_settings(t_values: List[int]) -> Dict:
    """ common settings for all violin plots
    :t_values: list of values on the x axis
    """
    if len(t_values) >= 2:
        dist = t_values[1] - t_values[0]
    else:
        dist = 1
    # dirty hack to make the violins fit into the plot area horizontally
    dist = 1 if dist==2 else dist
    return {"widths": 0.7 * dist}

def extract_metrics_from_json(json_data):
    """Process DKG data from JSON reports into a DataFrame"""
    rows = []
    for experiment in json_data:
        number_of_nodes = experiment['number_of_nodes']
        threshold = experiment.get('threshold', 2)  # Assuming threshold is provided, defaulting to 2
        
        for report in experiment['reports']:
            total_duration = parse_duration(report['total_duration'])  # Parse string to ms
            computation_time = parse_duration(report['computation'])
            io_time = parse_duration(report['io']['total_io'])
            
            for round_data in report['rounds']:
                round_name = round_data['round_name']
                round_duration = parse_duration(round_data['total_duration'])
                
                for stage in round_data.get('stages', []):
                    stage_name = stage['stage_name']
                    stage_duration = parse_duration(stage['duration'])
                    
                    # Append data for each stage
                    rows.append({
                        'n': number_of_nodes,
                        't': threshold,
                        'round': round_name,
                        'stage': stage_name,
                        'stage_duration_ms': stage_duration,
                        'round_duration_ms': round_duration,
                        'computation_ms': computation_time,
                        'io_ms': io_time,
                        'total_duration_ms': total_duration
                    })
    
    df = pd.DataFrame(rows)
    return df

def parse_duration(duration_str: str) -> float:
    """
    Parse the duration string, extract the numeric value, and convert it to milliseconds.
    Supports durations in ms, µs, and ns, with optional percentage suffix.

    Args:
    - duration_str: The string containing a duration, e.g., "38.39ms (85.8%)" or "6.88ms"
    
    Returns:
    - The duration converted to milliseconds (float).
    """
    # Regular expression to capture the numeric part and the time unit (ms, µs, ns)
    match = re.search(r"([0-9.]+)\s*(ms|µs|ns)?", duration_str)
    if match:
        value = float(match.group(1))
        unit = match.group(2)
        
        # Convert to milliseconds
        if unit == "µs":
            return value / 1000.0  # microseconds to milliseconds
        elif unit == "ns":
            return value / 1_000_000.0  # nanoseconds to milliseconds
        elif unit == "ms" or unit is None:
            return value  # milliseconds
        else:
            raise ValueError(f"Unknown time unit: {unit}")
    else:
        raise ValueError(f"Unable to parse duration: {duration_str}")

class Plotter():
    import matplotlib.pyplot as plt # this sets self.plt
    
    def __init__(self, output_dir, pos_alloc_file: str = None, write = False):
        """ :write: write graphs to file
        :pos_alloc_file: path to pos' allocation.json 
        """
        self.num_threads_dut = None
        self.pos_alloc_info = None
        if pos_alloc_file is None: # ugly:
            print_warn("Missing infos on allocation!")
        else:
            with open(pos_alloc_file, 'r') as f:
                self.pos_alloc_info = json.load(f)
            alloc_dir = os.path.join(os.path.dirname(pos_alloc_file), "..")
            dut_node = None
            for node in self.pos_alloc_info["nodes"]:
                with os.scandir(os.path.join(alloc_dir, node)) as it:
                    is_dut_node = True
                    for entry in it:
                        if entry.name.endswith(".csv"): # DuT node running the peers hasn't CSV files
                            is_dut_node = False
                            break
                    if not is_dut_node:
                        continue
                    dut_node = node
                    break

            lshw_path = os.path.join(alloc_dir, dut_node, "json+lshw.stdout")
            if not os.path.exists(lshw_path):
                print_warn("lshw missing, no num_threads shown!")
            else:
                with open(lshw_path, 'r') as f:
                    # fix broken JSON
                    data = f.read()
                    # fix first & last bytes
                    data = re.sub("^\[\n", '', data)
                    data = re.sub("\]\n\n$", '}', data)
                    # other broken stuff
                    data = re.sub("\n\s*}\s*{\s*\n", '\n }, "children" : [ {\n', data)
                    data = re.sub('"        {', '", "children" : [        {', data)
                    data = re.sub('0            {', ', "children" : [        {', data)
                    data = re.sub("\n\n", '\n ] \n', data)
                    data = re.sub("},\s*\n\s*]", '}\n]', data)
                    
                    lshw = json.loads(data)
                    
                    for item in lshw["children"][0]["children"]:
                        if item["class"] == "processor":
                            if re.match("cpu(:\d+)?", item["id"]) is None:
                                print_warn(f"Processor id has unknown format {item['id']}")
                            if self.num_threads_dut is None:
                                self.num_threads_dut = int(item["configuration"]["threads"])
                            else:
                                self.num_threads_dut += int(item["configuration"]["threads"])
            print(f"DuT {dut_node} has {self.num_threads_dut} threads")
        # values in inch
        # marging between figure and edge
        self.marginleft = 0.001

        self.output_dir = None
        if write:
            self.output_dir = output_dir
            # mpl.use("pgf") # comment out for debugging and development
            # # Need to patch backend because sketched lines not shown in pdf
            # #mpl.use("module://patched_backend_pgf.backend_pgf")
            # mpl.rcParams.update(self.pgf_with_custom_preamble)
        # self.filetypes = [".pgf", ".pdf"]
        self.filetypes = [".pdf"]
        self.grid_minor_color = "0.93"

    def config_x_axis_as_number_of_nodes(self, n_values: List[int]):
        if len(n_values) <= 32:
            self.ax.set_xticks(n_values)
            self.ax.set_xlim(min(n_values)-0.5, max(n_values)+0.5)
        else:
            print_warn("too many xticks, TODO")
        if self.num_threads_dut is not None:
            self.ax.axvline(self.num_threads_dut, ls='--', in_layout=False, color='#4a4a4a', zorder=2.00001)
        self.ax.set_xlabel("Number of Nodes $n$")

    def config_x_axis_as_threshold(self, t_values: List[int]):
        if len(t_values) <= 32:
            self.ax.set_xticks(t_values)
            self.ax.set_xlim(min(t_values)-0.4, max(t_values)+0.4)
        else:
            print_warn("too many xticks, TODO")
        self.ax.set_xlabel("Threshold $t$")

    def config_y_axis_as_time_in_ms(self):
        self.ax.set_ylabel("Time [ms]")
        self.ax.set_ylim(bottom=0)
        self.ax.yaxis.grid(True, which='major')
        self.ax.yaxis.grid(True, which='minor', color="0.8", alpha=0.45)
        # major grid is zorder=2, but minor grid can't be in the back: github.com/matplotlib/matplotlib/issues/5045
        # work around by setting alpha instead of self.grid_minor_color

        y_values = self.ax.yaxis.get_majorticklocs()
        if len(y_values) > 1:
            # use crude heuristic to add subgrid lines for each second
            if int(y_values[1] - y_values[0]) == 2000:
                self.ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(n=2))
            else:
                self.ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    def plot_dkg(self, dfraw: DataFrame) -> (str, str):
        """ returns (diagram label/caption, filename_root)"""
        # compute min, mean, max
        dfa = dfraw.groupby(level=["n","t"]).agg([np.min, np.mean, np.max])
        dfa.rename(columns={"amin": "min", "amax": "max"}, inplace=True)
        
        handles = {}; color_index = 0
        t_values = dfa.index.get_level_values('t').unique()
        # ugly hack to avoid single violin/point in plot
        max_n = max(dfa.index.get_level_values('n').unique())
        if max_n == max(t_values):
            t_values = list(t_values)
            t_values.remove(max_n)

        if len(t_values) > 4:
            iterrange = sample_t_values(t_values, num_samples=4)
        else:
            iterrange = t_values
        for t in iterrange:            
            df = dfa.loc[(dfa.index.get_level_values('t') == t)]["total_duration_ms"]
            min_values = df["min"]
            max_values = df["max"]
            x = df.index.get_level_values('n')
            self.ax.fill_between(x, min_values, max_values, color=colors[color_index], alpha=0.3, zorder=2.4)
            
            y = df["mean"]
            handle = self.ax.plot(x, y, marker=".", color=colors[color_index], zorder=2.5)[0]
            handles.setdefault(t, handle)
            color_index += 1
        
        labeldata = sorted(handles.keys(), reverse=True)
        labels = list(map(lambda t: f"t={t}", labeldata))
        handles = list(map(lambda x: handles[x], labeldata))
        self.ax.legend(handles, labels, loc="upper left")
        self.config_x_axis_as_number_of_nodes(dfraw.index.get_level_values('n').unique().values)
        self.config_y_axis_as_time_in_ms()
        
        return ("DKG", "dkg")

    def plot_dkg_violinplot(self, dfraw: DataFrame) -> (str, str):
        """ returns (diagram label/caption, filename_root)"""
        # Aggregating duplicates to avoid unstack issues
        dfa = dfraw.groupby(level=["n", "t", "round", "stage"]).mean()

        handles = {}; color_index = 0
        t_values = dfa.index.get_level_values('t').unique()
        # Avoid single violin/point in plot
        max_n = max(dfa.index.get_level_values('n').unique())
        if max_n == max(t_values):
            t_values = list(t_values)
            t_values.remove(max_n)

        if len(t_values) > 4:
            iterrange = sample_t_values(t_values, num_samples=4)
        else:
            iterrange = t_values

        for t in iterrange:
            t_df = dfa.loc[(dfa.index.get_level_values('t') == t)]
            # Ensure we have unique indices before unstacking
            t_df = t_df.groupby(level=["n"]).mean()
            n_as_column = t_df.unstack(level="n")
            x = t_df.index.get_level_values('n').unique().values
            handle_violin = self.ax.violinplot(n_as_column.values, positions=x, **violin_settings(x))
            color_violinplot_handle(handle_violin, colors[color_index])

            self.ax.scatter(x, n_as_column.agg([np.mean]), marker='o', color=colors[color_index], zorder=3)  # means

            for pc in handle_violin["bodies"]:
                pc.set_alpha(0.4)  # 0.3
                pc.set_zorder(2.4)
            for item in ["cmaxes", "cmins", "cbars"]:
                handle_violin[item].set_zorder(2.5)

            handles.setdefault(t, mpatches.Patch(color=colors[color_index], label=f"$t={t}$"))
            color_index += 1

        labeldata = sorted(handles.keys(), reverse=True)
        handles = list(map(lambda x: handles[x], labeldata))
        self.plt.legend(handles=handles, loc='upper left')

        self.config_x_axis_as_number_of_nodes(dfraw.index.get_level_values('n').unique().values)
        self.config_y_axis_as_time_in_ms()
        return ("DKG", "dkg_violin")

    def plot_dkg_inc_t_violinplot(self, dfraw: DataFrame) -> (str, str):
        """ returns (diagram label/caption, filename_root)"""
        # Aggregating duplicates to avoid unstack issues
        dfa = dfraw.groupby(level=["n", "t", "round", "stage"]).mean()

        handles = {}; color_index = 0
        n_values = dfa.index.get_level_values('n').unique()

        # Only plot for the last `n` value
        indices = [len(n_values) - 1]

        for n_index in indices:
            n = n_values[n_index]

            n_df = dfa.loc[(dfa.index.get_level_values('n') == n)]
            # Ensure we have unique indices before unstacking
            n_df = n_df.groupby(level=["t"]).mean()
            t_as_column = n_df.unstack(level="t")

            x = n_df.index.get_level_values('t').unique().values
            handle_violin = self.ax.violinplot(t_as_column.values, positions=x, **violin_settings(x))
            color_violinplot_handle(handle_violin, colors[color_index])

            self.ax.scatter(x, t_as_column.agg([np.mean]), marker='o', color=colors[color_index], zorder=3)  # means

            for pc in handle_violin["bodies"]:
                pc.set_alpha(0.4)  # 0.3
                pc.set_zorder(2.4)
            for item in ["cmaxes", "cmins", "cbars"]:
                handle_violin[item].set_zorder(2.5)

            handles[n] = mpatches.Patch(color=colors[color_index], label=f"$n={n}$")
            color_index += 1

        labeldata = sorted(handles.keys(), reverse=True)
        handles = list(map(lambda x: handles[x], labeldata))
        self.plt.legend(handles=handles, loc='upper left')

        self.config_x_axis_as_threshold(dfraw.index.get_level_values('t').unique().values)
        self.config_y_axis_as_time_in_ms()

        if self.num_threads_dut is None:
            caption = "DKG"
        else:
            caption = f"DKG; max. parallel threads: ${self.num_threads_dut}$"
        return (caption, "dkg_inc_t_violin")
    
    def plot_dkg_throughput(self, dfraw: DataFrame) -> (str, str):
        """ returns (diagram label/caption, filename_root)"""
        # Calculate throughput as inverse of total duration
        dfraw['throughput'] = 1 / dfraw['total_duration_ms']

        # Compute min, mean, max of throughput
        dfa = dfraw.groupby(level=["n", "t"]).agg([np.min, np.mean, np.max])
        dfa.rename(columns={"amin": "min", "amax": "max"}, inplace=True)

        handles = {}
        color_index = 0
        t_values = dfa.index.get_level_values('t').unique()
        
        # Select a range of `t` values to plot
        if len(t_values) > 4:
            iterrange = sample_t_values(t_values, num_samples=4)
        else:
            iterrange = t_values
        
        for t in iterrange:
            df = dfa.loc[(dfa.index.get_level_values('t') == t)]["throughput"]
            min_values = df["min"]
            max_values = df["max"]
            x = df.index.get_level_values('n')
            self.ax.fill_between(x, min_values, max_values, color=colors[color_index], alpha=0.3, zorder=2.4)
            
            y = df["mean"]
            handle = self.ax.plot(x, y, marker=".", color=colors[color_index], zorder=2.5)[0]
            handles.setdefault(t, handle)
            color_index += 1

        # Labeling the plot
        labeldata = sorted(handles.keys(), reverse=True)
        labels = list(map(lambda t: f"t={t}", labeldata))
        handles = list(map(lambda x: handles[x], labeldata))
        self.ax.legend(handles, labels, loc="upper left")

        self.config_x_axis_as_number_of_nodes(dfraw.index.get_level_values('n').unique().values)
        self.ax.set_ylabel("Throughput (1/ms)")  # Set y-axis label as throughput
        self.ax.set_ylim(bottom=0)
        self.ax.yaxis.grid(True, which='major')
        self.ax.yaxis.grid(True, which='minor', color="0.8", alpha=0.45)

        return ("DKG Throughput", "dkg_throughput")

    def plot_preprocess_inc_n(self, dfraw: DataFrame) -> (str, str):
        """ returns (diagram label/caption, filename_root)"""
        levels = list(dfraw.index.names); levels.remove("run_id") # compute min, mean, max, only aggregate over run_id
        dfa = dfraw.groupby(level=levels).agg([np.min, np.mean, np.max])
        dfa.rename(columns={"amin": "min", "amax": "max"}, inplace=True)

        handles = {}
        
        num_noncepair_values = dfa.index.get_level_values('num_noncepairs').unique()
        for (i, num_noncepair_index) in enumerate(sample_in_list(len(num_noncepair_values), num_samples=10)):
            num_noncepairs = num_noncepair_values[num_noncepair_index]
            df_nn = dfa.loc[(dfa.index.get_level_values('num_noncepairs') == num_noncepairs)]["duration_ms"]

            samples_t = 1
            t_values = df_nn.index.get_level_values('t').unique()
            for t in sample_t_values(t_values, samples_t):
                df = df_nn.loc[(df_nn.index.get_level_values('t') == t)]

                min_values = df["min"]
                max_values = df["max"]
                x = df.index.get_level_values('n')
                handle_area = self.ax.fill_between(x, min_values, max_values, alpha=0.3, zorder=2.4) # color=colors[i],

                y = df["mean"]
                handle_line = self.plt.plot(x, y, marker=".", color=colors[i], zorder=2.5)[0]
                handles[(num_noncepairs, t)] = [handle_line, handle_area]

        color_handles(handles.values())
        labeldata = sorted(handles.keys(), reverse=True)
        labels = list(map(lambda x: f"$noncepairs={x[0]}, t={x[1]}$", labeldata))
        handles = list(map(lambda x: handles[x][0], labeldata))
        self.plt.legend(handles, labels, loc='best')

        self.config_x_axis_as_number_of_nodes(dfraw.index.get_level_values('n').unique().values)
        self.config_y_axis_as_time_in_ms()

        return ("Preprocessing", f"inc_n_prep_{get_range_params(list(map(lambda x: x[0], labeldata)))}")
    
    def plot_preprocess_inc_n_violinplot(self, dfraw: DataFrame) -> (str, str):
        """ returns (diagram label/caption, filename_root)"""
        dfa = dfraw.groupby(level=["n","t"]).apply(lambda x: x)
        
        handles = {}
        
        t_values = dfa.index.get_level_values('t').unique()
        for t in sample_t_values(t_values, 2): # don't plot all value of "t"
            t_df = dfa.loc[(dfa.index.get_level_values('t') == t)]
            
            num_noncepair_values = t_df.index.get_level_values('num_noncepairs').unique()
            for num_noncepair_index in sample_in_list(len(num_noncepair_values), num_samples=4):
                num_noncepairs = num_noncepair_values[num_noncepair_index]

                df = t_df.loc[(t_df.index.get_level_values('num_noncepairs') == num_noncepairs)]["duration_ms"]
                
                n_as_column = df.unstack(level="n")
                x = df.index.get_level_values('n').unique().values

                handle_violin = self.ax.violinplot(n_as_column.values, positions=x, **violin_settings(x))
                handle_mean = self.ax.scatter(x, n_as_column.agg([np.mean]), marker='o', zorder=3)
                
                for pc in handle_violin["bodies"]:
                    #pc.set_facecolor(colors[color_index])
                    pc.set_alpha(0.4) # 0.3
                    pc.set_zorder(2.4)
                handles_violin_components = []
                for item in ["cmaxes", "cmins", "cbars"]:
                    handle_violin[item].set_zorder(2.5)
                    handles_violin_components.append(handle_violin[item])
                handle_legend = mpatches.Patch(label=f"$noncepairs={num_noncepairs}, t={t}$")
                handles[(num_noncepairs,t)] = [handle_legend, handle_mean] + handle_violin["bodies"] + handles_violin_components

        labeldata = sorted(handles.keys(), reverse=True)
        handles = list(map(lambda x: handles[x], labeldata))
        color_handles(reversed(handles))
        self.plt.legend(handles=list(map(lambda l: l[0], handles)), loc='best')
        
        self.config_x_axis_as_number_of_nodes(dfraw.index.get_level_values('n').unique().values)
        self.config_y_axis_as_time_in_ms()
        
        return ("Preprocessing", "inc_n_prep_violin")
    
    def plot_preprocess_inc_nn(self, dfraw: DataFrame) -> (str, str):
        """ returns (diagram label/caption, filename_root)"""
        levels = list(dfraw.index.names); levels.remove("run_id") # compute min, mean, max, only aggregate over run_id
        dfa = dfraw.groupby(level=levels).agg([np.min, np.mean, np.max])
        dfa.rename(columns={"amin": "min", "amax": "max"}, inplace=True)

        handles = {};
        
        n_values = dfa.index.get_level_values('n').unique()
        for n_index in sample_in_list(len(n_values), num_samples=4):
            n = n_values[n_index]
            df_n = dfa.loc[(dfa.index.get_level_values('n') == n)]["duration_ms"]
            
            t_values = df_n.index.get_level_values('t').unique()
            if len(t_values) > 1:
                iterrange = sample_t_values(t_values, num_samples=1)
            else:
                iterrange = t_values

            for t in iterrange:
                df = df_n.loc[(df_n.index.get_level_values('t') == t)]

                min_values = df["min"]
                max_values = df["max"]
                x = df.index.get_level_values('num_noncepairs')
                handle_area = self.ax.fill_between(x, min_values, max_values, alpha=0.3)

                y = df["mean"]
                handle = self.plt.plot(x, y, marker=".", label=f"$n={n}, t={t}$")[0]
                handles[(n, t)] = [handle, handle_area]
        
        color_handles(handles.values())
        labeldata = sorted(handles.keys(), reverse=True)
        handles = list(map(lambda x: handles[x][0], labeldata))
        self.plt.legend(handles=handles, loc='upper left')
        
        self.ax.set_xlim(left=0)
        self.ax.set_xlabel("Number of nonce pairs $\\nu$")
        self.config_y_axis_as_time_in_ms()
        
        return ("Preprocessing", f"inc_nn_prep")
    
    def plot_sign_inc_msglen(self, dfraw: DataFrame) -> (str, str):
        """ returns (diagram label/caption, filename_root)"""
        # compute min, mean, max
        levels = list(dfraw.index.names); levels.remove("run_id") # only aggregate over run_id
        dfa = dfraw.groupby(level=levels).agg([np.min, np.mean, np.max])
        dfa.rename(columns={"amin": "min", "amax": "max"}, inplace=True)
        
        handles = {}
        
        n_values = dfa.index.get_level_values('n').unique()
        # Set color later because this has the disadvantage of requiring the number of colors beforehand
        #self.ax.set_prop_cycle(color=color_cycles[len(indices)]) # must be set before plt.plot().

        for n_index in sample_in_list(len(n_values), num_samples=4):
            n = n_values[n_index]            
            df_n = dfa.loc[(dfa.index.get_level_values('n') == n)]["duration_ms"]

            for t in sample_t_values(df_n.index.get_level_values('t').unique(), num_samples=2):
                df = df_n.loc[(df_n.index.get_level_values('t') == t)]

                min_values = df["min"]
                max_values = df["max"]
                x = df.index.get_level_values('msg_len')
                handle_area = self.ax.fill_between(x, min_values, max_values, alpha=0.3, zorder=2.4)
                
                y = df["mean"]
                handle = self.ax.plot(x, y, marker=".", zorder=2.5)[0]
                handles[(n, t)] = [handle, handle_area]

        color_handles(handles.values())
        labeldata = sorted(handles.keys(), reverse=True)
        labels = list(map(lambda x: f"$n={x[0]}, t={x[1]}$", labeldata))
        handles = list(map(lambda x: handles[x][0], labeldata))
        self.plt.legend(handles, labels, loc='best')
        x_values = dfraw.index.get_level_values('msg_len').unique().values
        x_max = x_values[-1]
        x_min = x_values[0]
        self.ax.set_xlim(left=0, right=x_max+x_min)
        self.ax.set_xlabel("Message Length [B]")
        self.config_y_axis_as_time_in_ms()
        
        return ("Signing", "inc_msglen_sign")
    
    def plot_sign_inc_n_violinplot(self, dfraw: DataFrame) -> (str, str):
        """ returns (diagram label/caption, filename_root)"""
        dfa = dfraw.groupby(level=["n","t"]).apply(lambda x: x)

        handles = {}

        t_values = dfa.index.get_level_values('t').unique()
        if 2 in t_values:
            t_values = [2]
        else:
            t_values = sample_t_values(t_values, num_samples = 1) # don't plot all value of "t"

        for t in t_values:
            #if t == 5:
            #    continue
            t_df = dfa.loc[(dfa.index.get_level_values('t') == t)]
            
            msg_len_values = t_df.index.get_level_values('msg_len').unique()
            for (color_index, msg_len_index) in enumerate(sample_in_list(len(msg_len_values), num_samples=2)):
                msg_len = msg_len_values[msg_len_index]

                df = t_df.loc[(t_df.index.get_level_values('msg_len') == msg_len)]["duration_ms"]
                
                n_as_column = df.unstack(level="n")
                x = df.index.get_level_values('n').unique().values

                handle = self.ax.violinplot(n_as_column.values, positions=x, **violin_settings(x))
                self.ax.scatter(x, n_as_column.agg([np.mean]), marker='o', color=colors[color_index], zorder=3) # means
                
                for pc in handle["bodies"]:
                    pc.set_facecolor(colors[color_index])
                    pc.set_alpha(0.4) # 0.3
                    pc.set_zorder(2.4)
                for item in ["cmaxes", "cmins", "cbars"]:
                    handle[item].set_color(colors[color_index])
                    handle[item].set_zorder(2.5)
                handles[(t,msg_len)] = mpatches.Patch(color=colors[color_index], label=f"$msg\_len=$ \\SI{{{msg_len}}}{{\\byte}}$, t={t}$")
        
        labeldata = sorted(handles.keys(), reverse=True)
        handles = list(map(lambda x: handles[x], labeldata))
        self.plt.legend(handles=handles, loc='upper left')

        self.config_x_axis_as_number_of_nodes(dfraw.index.get_level_values('n').unique().values)
        self.config_y_axis_as_time_in_ms()
        
        return ("Signing", "inc_n_sign_violin")
    
    def plot_sign(self, dfraw: DataFrame) -> (str, str):
        """ returns (diagram label/caption, filename_root)"""
        # compute min, mean, max
        levels = list(dfraw.index.names); levels.remove("run_id") # only aggregate over run_id
        dfa = dfraw.groupby(level=levels).agg([np.min, np.mean, np.max])
        dfa.rename(columns={"amin": "min", "amax": "max"}, inplace=True)
        
        handles = {}
        
        t_values = dfa.index.get_level_values('t').unique()
        if 2 in t_values:
            t_values = [2]
        else:
            t_values = sample_t_values(t_values, num_samples = 1) # don't plot all value of "t"
        for t in t_values:
            t_df = dfa.loc[(dfa.index.get_level_values('t') == t)]

            samples_n = 2 # don't plot all value of "n"
            # increasing n
            msg_len_values = t_df.index.get_level_values('msg_len').unique()
            for (color_index, msg_len_index) in enumerate(sample_in_list(len(msg_len_values), num_samples=samples_n)):
                msg_len = msg_len_values[msg_len_index]
                
                df = t_df.loc[(t_df.index.get_level_values('msg_len') == msg_len)]["duration_ms"]
                min_values = df["min"]
                max_values = df["max"]
                x = df.index.get_level_values('n')
                self.ax.fill_between(x, min_values, max_values, color=colors[color_index], alpha=0.3, zorder=2.4)
                
                y = df["mean"]
                handle = self.ax.plot(x, y, marker=".", color=colors[color_index], zorder=2.5)[0]
                handles[msg_len] = (handle, t)
        
        labeldata = sorted(handles.keys(), reverse=True)
        labels = list(map(lambda x: f"$msg\_len=$ \\SI{{{x}}}{{\\byte}}$, t={handles[x][1]}$", labeldata))
        handles = list(map(lambda x: handles[x][0], labeldata))
        self.plt.legend(handles, labels, loc='upper left') # instead of "best" because axvline()s are not ignored

        self.config_x_axis_as_number_of_nodes(dfraw.index.get_level_values('n').unique().values)
        self.config_y_axis_as_time_in_ms()
        
        return ("Signing", "inc_n_sign")

    def plot_sign_inc_t_violinplot(self, dfraw: DataFrame) -> (str, str):
        """ returns (diagram label/caption, filename_root)"""
        dfa = dfraw.groupby(level=["n","t"]).apply(lambda x: x)

        handles = {};
        n_values = dfa.index.get_level_values('n').unique()
        if len(n_values) > 1 and self.num_threads_dut in n_values:
            n_values = [self.num_threads_dut, n_values[-1:][0]]
        else:
            indices = sample_in_list(len(n_values), num_samples=4)
            n_values = list(map(lambda i: n_values[i], indices))
            # TODO better filter

        for n in n_values:
            n_df = dfa.loc[(dfa.index.get_level_values('n') == n)]
            
            msg_len_values = n_df.index.get_level_values('msg_len').unique()
            for msg_len_index in sample_in_list(len(msg_len_values), num_samples=2):
                msg_len = msg_len_values[msg_len_index]

                df = n_df.loc[(n_df.index.get_level_values('msg_len') == msg_len)]["duration_ms"]
                t_as_column = df.unstack(level="t")
                x = df.index.get_level_values('t').unique().values

                handle_violin = self.ax.violinplot(t_as_column.values, positions=x, **violin_settings(x))
                handle_mean = self.ax.scatter(x, t_as_column.agg([np.mean]), marker='o', zorder=3)

                for pc in handle_violin["bodies"]:
                    pc.set_alpha(0.4) # 0.3
                    pc.set_zorder(2.4)
                handles_violin_components = []
                for item in ["cmaxes", "cmins", "cbars"]:
                    handle_violin[item].set_zorder(2.5)
                    handles_violin_components.append(handle_violin[item])
                handle_legend = mpatches.Patch(label=f"$n={n}, msg\_len=$ \\SI{{{msg_len}}}{{\\byte}}")
                handles[(n,msg_len)] = [handle_legend, handle_mean] + handle_violin["bodies"] + handles_violin_components

        color_handles(handles.values())
        labeldata = sorted(handles.keys(), reverse=True)
        handles = list(map(lambda x: handles[x][0], labeldata))
        self.plt.legend(handles=handles, loc='lower right')

        self.config_x_axis_as_threshold(dfraw.index.get_level_values('t').unique().values)
        self.config_y_axis_as_time_in_ms()

        if self.num_threads_dut is None:
            caption = "Signing"
        else:
            caption = f"Signing; max. parallel threads: ${self.num_threads_dut}$"
        return (caption, "inc_t_sign_violin")

    def __new_plot(self, break_y_axis = None, figsize = None):
        """
            figsize: (width, height)
        """
        if break_y_axis is None:
            self.fig, self.ax = self.plt.subplots()
            self.ax.grid(color="0.8")
        else:
            assert(break_y_axis > 1)
            # see https://matplotlib.org/stable/gallery/subplots_axes_and_figures/broken_axis.html
            self.fig, axes = self.plt.subplots(break_y_axis, 1, sharex=True, figsize=figsize)
            self.axs = list(reversed(axes)) # self.axs contains axes from bottom to top
            self.fig.subplots_adjust(hspace=0.05)  # adjust space between axes

            # joint X grid
            self.ax_grid = self.fig.add_subplot(111, zorder=-1)
            for _, spine in self.ax_grid.spines.items():
                spine.set_visible(False)
            self.ax_grid.tick_params(labelleft=False, labelbottom=False, left=False, right=False)
            self.ax_grid.get_shared_x_axes().join(self.ax_grid, *self.axs) #self.axs[-1])
            self.ax_grid.grid(axis="x", color="0.8")

            for i in range(0, break_y_axis-1):
                prevax = i
                nextax = i+1
                # hide the spines between x axes
                #self.axs[nextax].spines["bottom"].set_visible(False)
                #self.axs[prevax].spines["top"].set_visible(False)
                # draw wiggled line
                for spine in [self.axs[nextax].spines["bottom"], self.axs[prevax].spines["top"]]:
                    spine.set_edgecolor("0.5")
                    spine.set_zorder(1.9)
                    spine.set_sketch_params(scale=5, length=10, randomness=42)
                self.axs[nextax].xaxis.tick_top()
                self.axs[nextax].tick_params(labeltop=False, length=0) # no tick labels at the top

                # cut-out indicator
                d = .6  # proportion of vertical to horizontal extent of the slanted line
                kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color='k', mec='k', mew=1, 
                              clip_on=False)
                self.axs[prevax].plot([0, 1], [1, 1], transform=self.axs[prevax].transAxes, **kwargs)
                self.axs[nextax].plot([0, 1], [0, 0], transform=self.axs[nextax].transAxes, **kwargs)

            for ax in self.axs:
                ax.grid(color="0.8")

    def __finalize_plot(self, max_width = 5.5, bottomlabel: str = None, labelmarginbottom=0.01, leftlabel: str = None, leftlabel_h = 0.5, labelmarginleft=0.005, marginbottom = 0.1):
        """
        :labelmarginleft:, :labelmarginbottom:   marging between text center and edge
        """
        width, height = self.fig.get_size_inches()
        w = labelmarginleft / width
        h = labelmarginbottom / height
        if bottomlabel is not None:
            self.fig.text(0.5, h, bottomlabel, va='bottom', ha='center', transform=self.fig.transFigure)

        if leftlabel is not None:
            self.fig.text(w, leftlabel_h, leftlabel, va='center', ha='left', rotation=90, transform=self.fig.transFigure)

        # last 2 entries for a smaller right/top margins
        self.plt.tight_layout(rect=((width+self.marginleft)/width-1, (height+marginbottom)/height-1, (width+0.18)/width, (height+0.18)/height)) # (left, bottom, right, top) relative to the whole figure

        #self.fig.set_size_inches(7.25, 4.5) # (width, height)
        width, height = self.fig.get_size_inches()
        if width > max_width :#7.25:
            print_warn("Warn: limiting width")
            self.fig.set_figwidth(max_width)

        if hasattr(self, "ax") and self.output_dir is not None:
            self.ax.set_xticklabels([f"{int(x)}" for x in self.ax.get_xticks()])
            self.ax.set_yticklabels([f"{int(y)}" for y in self.ax.get_yticks()])

    def __save_plot(self, filename_root: str):
        if self.pos_alloc_info is None:
            datestring = "Unknown_date"
        else:
            parsed_date = datetime.datetime.strptime(self.pos_alloc_info["created"], "%Y-%m-%d %H:%M:%S.%f") # alloc date
            datestring = parsed_date.strftime("%Y-%m-%d_%H%M%S")

        if self.output_dir is None:
            self.plt.show()
        else:
            for fileEnding in self.filetypes:
                filepath = os.path.join(self.output_dir, f"{datestring}_{filename_root}{fileEnding}")
                self.plt.savefig(filepath, transparent=True)
                print(f"Wrote {filepath}")

    def plot(self, resultDirs: str):
        """ 
        :resultDir: the folder with the csv files
        """
        nmap = {}
        testcase_desc = None
        testcase = None
        for resultDir in resultDirs:
            for resultFile in glob.iglob(os.path.join(os.path.abspath(resultDir), "n*_t*_*.csv")):
                if os.path.isfile(resultFile):
                    fileName = os.path.basename(resultFile)
                    
                    nGroup = "n"
                    tGroup = "t"
                    testcaseGroup = "testcase"
                    
                    fileNameMatches = re.search('^n(?P<%s>[0-9]+)_t(?P<%s>[0-9]+)_(?P<%s>\S+)(\.csv)$' % (nGroup, tGroup, testcaseGroup), fileName)
                    print(fileNameMatches)
                    if fileNameMatches is None:
                        raise ValueError("Unknown file {} present".format(fileName))
                        
                    n = int(fileNameMatches.group(nGroup))
                    t = int(fileNameMatches.group(tGroup))
                    testcase_desc_parsed = fileNameMatches.group(testcaseGroup)
                    testcase_parsed, _ = parseTestcaseDesc(testcase_desc_parsed)
                    
                    if testcase is None:
                        testcase = testcase_parsed
                    else:
                        assert(testcase == testcase_parsed)
                    #print(f"parsed n={n}, t={t}, testcase={testcase_desc_parsed}")
                    df = pd.read_csv(resultFile, index_col="run_id", dtype={"is_ok": bool, "duration_ns": int }) # usecols=["is_ok", "run_id", "duration_ns"]

                    if "is_ok" in df.columns:
                        err = len(df.loc[df["is_ok"] == False])
                        if err != 0:
                            print_warn(f"{fileName}: Dropping {err} useless measurements")
                            df.drop(df[df["is_ok"] == False].index, inplace=True)
                        df.drop(["is_ok"], axis=1, inplace=True)
                    # convert nanoseconds to milliseconds
                    df['duration_ms'] = df['duration_ns'].apply(nsToMs)
                    df.drop(["duration_ns"], axis=1, inplace=True)

                    mytmap = nmap.setdefault(n, {})
                    testcasemap = mytmap.setdefault(t, {})
                    testcasemap[testcase_desc_parsed] = df
        # sys.exit(0)
        dfs = []
        ns = []
        
        for n, tmap in nmap.items():
            tmapresults = {}
            for t, testcasemap in tmap.items(): # this could be a map() call
                dataframes = []
                for testcase_desc, dataframe in testcasemap.items():
                    testcase, testcase_args = parseTestcaseDesc(testcase_desc)
                    if testcase == "dkg":
                        pass  # nothing to be done
                    elif testcase == "preprocess": # add num_noncepairs as index level
                        dataframe = pd.concat([dataframe], keys=[testcase_args["num_noncepairs"]], names=["num_noncepairs"])
                    elif testcase == "sign": # add msg_len is index level
                        dataframe = pd.concat([dataframe], keys=[testcase_args["msg_len"]], names=["msg_len"])
                    else:
                        assert(False)
                    dataframes.append(dataframe)
                
                tmapresults[t] = pd.concat(dataframes)
            
            dfs.append(pd.concat(tmapresults, names=["t"])) # keys (t values) are automatically used as keys in the DF
            ns.append(n)
        
        dfraw = pd.concat(dfs, keys=ns, names=["n"])
        dfraw.sort_index(inplace=True)

        print(f"Parsed CSV for n={get_range_params(dfraw.index.get_level_values('n').unique())}, t={get_range_params(dfraw.index.get_level_values('t').unique())}, ", end='')
        if testcase == "dkg":
            print("dkg\n")
        elif testcase == "preprocess":
            print(f"preprocess num_noncepairs={get_range_params(dfraw.index.get_level_values('num_noncepairs').unique())}\n")
        elif testcase == "sign":
            print(f"sign msg_len={get_range_params(dfraw.index.get_level_values('msg_len').unique())}\n")

        plotting_funcs = []
        
        if testcase == "dkg":
            plotting_funcs.append(self.plot_dkg)
            plotting_funcs.append(self.plot_dkg_violinplot)
            plotting_funcs.append(self.plot_dkg_inc_t_violinplot)
        elif testcase == "preprocess":
            plotting_funcs.append(self.plot_preprocess_inc_n)
            plotting_funcs.append(self.plot_preprocess_inc_nn)
            plotting_funcs.append(self.plot_preprocess_inc_n_violinplot)
        elif testcase == "sign":
            plotting_funcs.append(self.plot_sign)
            plotting_funcs.append(self.plot_sign_inc_n_violinplot)
            plotting_funcs.append(self.plot_sign_inc_msglen)
            plotting_funcs.append(self.plot_sign_inc_t_violinplot)
        else:
            assert(False)

        n_info = get_range_params(dfraw.index.get_level_values('n').unique())

        for func in plotting_funcs:
            self.__new_plot()
            bottomlabel, filename_root = func(dfraw)
            self.__finalize_plot(bottomlabel=bottomlabel)
            self.__save_plot(f"n{n_info}_{filename_root}")
    
    def plot_dkg_all(self, dfraw: pd.DataFrame):
        """ 
        Generate all DKG-related plots and save them to the specified directory.
        """
        # Define the list of plotting functions
        plotting_funcs = [
            self.plot_dkg,
            self.plot_dkg_violinplot,
            self.plot_dkg_inc_t_violinplot,
            self.plot_dkg_throughput
        ]

        # Get the range of 'n' values for filename formatting
        n_info = get_range_params(dfraw.index.get_level_values('n').unique())

        # Loop over each plotting function and save the plots
        for func in plotting_funcs:
            self.__new_plot()  # Create a new plot

            # Generate the plot and get the labels and filename
            bottomlabel, filename_root = func(dfraw)

            # Finalize the plot (add labels, adjust layout, etc.)
            self.__finalize_plot(bottomlabel=bottomlabel)

            # Save the plot to the output directory
            self.__save_plot(f"n{n_info}_{filename_root}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot results of benchmarks")
    parser.add_argument('-f', '--folder_scene', dest='folder_scene', help='Scene folder where the results directory is.', required=True)
    parser.add_argument("-d", "--dir", dest='dir', help='Playbook directory.', required=False)
    parser.add_argument("-s", "--scene", dest='scene', help='Scenario name.', required=False)

    if "argcomplete" in sys.modules:
        sys.modules.argcomplete.autocomplete(parser)
    args = parser.parse_args()
    resultDir = args.folder_scene
    path = os.path.realpath(resultDir)
    if not os.path.exists(path):
        print("Dir does not exist", file=sys.stderr)
        exit(1)
    pos_alloc_file = findConfigJsonHeuristic(path)
    if pos_alloc_file is None:
        print_warn("Could not find pos metadata json")
    output = resultDir
    print("Args")
    print(args)
    print("Result path")
    print(path)
    print("OUTPUT")
    print(output)
    print("POS_ALLOC_FILE")
    print(pos_alloc_file)
    json_data = parse_json_files(path)
    print("JSON DATA of files")
    print(json_data)

    # Process the parsed data
    dfraw = extract_metrics_from_json(json_data)
    dfraw.set_index(['n', 't', 'round', 'stage'], inplace=True)
    print("DFRAW")
    print(dfraw)
    p = Plotter(output, pos_alloc_file, write=(output is not None))
    p.plot_dkg_all(dfraw=dfraw)