import configparser,tempfile,sys,collections
import numpy as np, cv2
import compare_sheet as cs
import image_preprocessing_module as prep, table_and_cell_detection_model as det, transcription as tr
cfg=configparser.ConfigParser(); cfg.read(cs.REPO/"configuration.ini"); t=cfg["TableAndCellDetection"]
def contours(station,f):
    g,b,o=prep.image_preprocessing(str(cs.REPO/cfg["Directories"]["full_datadir"]/station/f))
    return det.table_and_cell_detection(g,b,o,station,f,tempfile.mkdtemp(),clip_up=int(t["clip_up"]),clip_down=int(t["clip_down"]),clip_left=int(t["clip_left"]),clip_right=int(t["clip_right"]),
      max_table_width=int(t["max_table_width"]),max_table_height=int(t["max_table_height"]),min_cell_width_threshold=int(t["min_cell_width_threshold"]),max_cell_width_threshold=int(t["max_cell_width_threshold"]),
      min_cell_height_threshold=int(t["min_cell_height_threshold"]),max_cell_height_threshold=int(t["max_cell_height_threshold"]),space_height_threshold=int(t["space_height_threshold"]),space_width_threshold=int(t["space_width_threshold"]),
      max_cell_height_per_box=int(t["max_cell_height_per_box"]),no_of_rows=int(t["no_of_rows"]),no_of_columns=int(t["no_of_columns"]))[0]
def gap_rows(cnts, n_rows, key="top"):
    """Cut the sorted y-positions at the (n_rows-1) largest gaps."""
    y=np.array([cv2.boundingRect(c)[1] + (cv2.boundingRect(c)[3]//2 if key=="mid" else 0) for c in cnts])
    order=np.argsort(y); ys=y[order]
    cuts=np.sort(np.argsort(np.diff(ys))[-(n_rows-1):])
    rows=[]; start=0
    for c in list(cuts+1)+[len(ys)]:
        rows.append([cnts[i] for i in order[start:c]]); start=c
    return rows
if __name__=="__main__":
    for s,f in [("601","601_196705_SF.JPG"),("905","905_197311_SF.JPG"),("508","508_198802_SF.JPG"),("203","203_196905_SF.JPG")]:
        cn=contours(s,f); n=tr.get_max_rows_from_filename(f)
        print(f"\n{f}: {len(cn)} contours, expected rows {n}")
        for name,rows in [("KMeans top (current)",tr.organize_contours_top(cn,f)),("KMeans mid (current)",tr.organize_contours_midpoint(cn,f)),("column-wise",__import__("row_grouping").column_wise_rows(cn,n))]:
            sizes=[len(r) for r in rows]; ok=sum(20<=x<=26 for x in sizes)
            print(f"  {name:<22} rows {len(rows):>3}  rows with 20-26 cells: {ok:>2}   min {min(sizes)} max {max(sizes)}  odd sizes {[x for x in sizes if not 20<=x<=26][:8]}")
