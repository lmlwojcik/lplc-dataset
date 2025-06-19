import json
import argparse
from bokeh.plotting import figure, save
#from bokeh.io import export_png
from bokeh.models.ranges import DataRange
import cv2

def main(config, log_file, results_file):
    with open(config, "r") as fd:
        cfg = json.load(fd)

    if log_file is not None:
        with open(log_file, "r") as fd:
            logs = json.load(fd)

        x_axis = [x['epoch'] for x in logs]

        for g in cfg['graphs']:
            fig = figure(**g['fig_args'], width=g['width_per_epoch']*len(x_axis))
            fig.y_range.start = 0

            for v,c in zip(g['values'], g['colors']):
                fig.line(x_axis,[x[v] for x in logs], line_width=2, color=c, legend_label=v)
            fig.legend.location = 'bottom_left'

            save(fig, filename=f"{cfg['save_dir']}/{g['fname']}.html", title=g['fig_args']['title'])
            if cfg['save_images']:
                fig.background_fill_color = None
                fig.border_fill_color = None
                #export_png(fig, f{cfg['save_dir']/g['name'].png})

    if results_file is not None:
        with open(results_file, "r") as fd:
            metrics = json.load(fd)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default="configs/config_graph.json")
    parser.add_argument('-l', '--log_file', default=None)
    parser.add_argument('-r', '--results_file', default=None)

    args = vars(parser.parse_args())
    
    main(**args)