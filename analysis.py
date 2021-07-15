import pandas as pd
import numpy as np
from pandas.core.algorithms import value_counts
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
from scipy.stats import mannwhitneyu, shapiro


def read_data():
    all_data = pd.DataFrame()
    for csv_file_name in glob.glob("./Willow_data/*.csv"):
        with open(csv_file_name, "r") as csv_file:
            tmp_df = pd.read_csv(csv_file, sep=";")
            file_stem_name = csv_file_name.split("/")[-1]

            if file_stem_name.startswith("LB"):
                soiltype = "loam"
            elif file_stem_name.startswith("SB"):
                soiltype = "sand"
            else:
                raise Exception("Can't identify soiltype! " + file_stem_name)

            sp = file_stem_name.split("_")
            plot_id = sp[0].replace("SB", "sand-").replace("LB", "loam-")

            if sp[1].startswith("2"):
                year = "2020"
            elif sp[1].startswith("3"):
                year = "2021"
            else:
                raise Exception("Can't identify year! " + file_stem_name)

            tmp_df["soiltype"] = soiltype
            tmp_df["plot_id"] = plot_id
            tmp_df["year"] = year
            tmp_df["Diameter"]=tmp_df["Diameter"].replace(["<0.5", "<0,5", ">0.5", "0", "0,4"], 0.25)
            tmp_df["Diameter"] = tmp_df.apply(
                lambda row: float(str(row["Diameter"]).replace(",", ".")), axis=1
            )
            tmp_df["Height"] = tmp_df.apply(
                lambda row: float(str(row["Height"]).replace(",", ".")), axis=1
            )
            tmp_df = tmp_df.astype({"Diameter": float, "Height": float})
            all_data = pd.concat([all_data, tmp_df])
    all_data.dropna(inplace=True)
    all_data.reset_index(inplace=True, drop=True)
    all_data["Height/Diameter"] = all_data.apply(
        lambda row: row["Height"] / row["Diameter"], axis=1
    )
    precise_data=all_data.copy()
    precise_data=precise_data[precise_data["Diameter"]>=0.5]
    return all_data, precise_data


class DataAnalyzer():
    def __init__(self, data, outdir):
        self.data=data
        self.outdir=outdir

def specific_scatter(data, category, years, prefix=""):
    ordered = list(data[category].unique())
    ordered.sort(key=lambda x: (x[:-1], x[-1]))
    pp = sns.pairplot(
        data,
        hue=category,
        diag_kws={"common_norm": False},
        plot_kws={"alpha": 0.3},
        hue_order=ordered,
    )
    pp.map_lower(sns.kdeplot, levels=[0.35], common_norm=False)
    pp.map_diag(sns.histplot, kde=True, common_norm=False, stat="density")
    plt.gcf().suptitle("+".join(years))
    plt.savefig(f"plots/{prefix}{category}_scatter_matrix_{'_'.join(years)}.png")
    plt.close("all")


def make_scatter_plots(data):
    for years in [["2020"], ["2021"], ["2020", "2021"]]:
        year_data = data[data["year"].isin(years)]
        specific_scatter(year_data, "soiltype", years)
        for s_type in ["sand", "loam"]:
            specific_scatter(
                year_data[year_data["soiltype"] == s_type],
                "plot_id",
                years,
                s_type + "_",
            )
            specific_scatter(
                year_data[year_data["soiltype"] == s_type], "Clone", years, s_type + "_"
            )


def make_box_plots(data):
    for variable in ["Height", "Diameter", "Height/Diameter"]:
        for years in [["2020"], ["2021"], ["2020", "2021"]]:
            year_data=data[data["year"].isin(years)]
            for x_var in ["soiltype", "plot_id"]:
                ordered = list(year_data[x_var].unique())
                ordered.sort(key=lambda x: (x[:-1], x[-1]))
                ax = sns.boxplot(
                    x=x_var,
                    y=variable,
                    data=year_data,
                    order=ordered,
                )
                ax.set_title("+".join(years))
                plt.savefig(
                    f"plots/boxplot_{x_var}_{'_'.join(years)}_{variable.replace('/','Over')}.png"
                )
                plt.close("all")
                if len(years)>1:
                    ax = sns.boxplot(
                        x=x_var,
                        y=variable,
                        data=year_data,
                        hue="year",
                        order=ordered,
                    )
                    ax.set_title("+".join(years))
                    plt.savefig(
                        f"plots/boxplot_{x_var}_{'_'.join(years)}_combined_{variable.replace('/','Over')}.png"
                    )
                    plt.close("all")


def make_p_value_norm_dist_plot(data):
    p_values = []
    for variable in ["Height", "Diameter", "Height/Diameter"]:
        for year in [["2020"], ["2021"]]:
            for uniq in data["plot_id"].unique():
                shapiro_results = shapiro(
                    data[(data["year"].isin(year)) & (data["plot_id"] == uniq)][
                        variable
                    ]
                )
                p_values.append(shapiro_results[1])
    ax = sns.histplot(p_values, bins=np.linspace(0, 1, 21))
    ax.set_xlabel("p-value")
    ax.set_ylabel("Count of plot measurements")
    ax.set_title("Shapiro test")
    plt.vlines([0.05], ymin=0, ymax=30, colors="r", label="0.05")
    plt.savefig("plots/p_values_norm_dist.png")
    plt.close("all")

def combined_pie_chart(data, years, prefix):
    fig, ax = plt.subplots()
    size = 0.5
    sand_plots=sorted([v for v in data[data["soiltype"]=="sand"]["plot_id"].unique()])
    loam_plots=sorted([v for v in data[data["soiltype"]=="loam"]["plot_id"].unique()])
    sand_values=[data[data["plot_id"]==v]["plot_id"].count() for v in sand_plots]
    loam_values=[data[data["plot_id"]==v]["plot_id"].count() for v in loam_plots]
    vals = [sum(sand_values), sum(loam_values)]
    sand_cmap = plt.get_cmap("Oranges")
    loam_cmap = plt.get_cmap("Blues")
    inner_colors = [sand_cmap(150), loam_cmap(150)]
    outer_colors = np.concatenate((sand_cmap([70, 90, 110, 130]),loam_cmap([90,110,130])))
    inner_wedges, _, _ = ax.pie(vals, radius=1.0-size, colors=inner_colors,
    textprops=dict(color="w", weight="bold") ,autopct=lambda pct: f"{pct:.1f}%\n({int(np.round(pct/100.*sum(vals),0))})")
    outer_wedges, outer_text, outer_num_text = ax.pie(sand_values + loam_values, radius=1.0, colors=outer_colors,
        wedgeprops=dict(width=size, edgecolor='w'), labels=[v.replace("loam","plot").replace("sand","plot") for v in sand_plots+loam_plots],
    textprops=dict(color="w", weight="bold") ,autopct=lambda pct: f"{pct:.1f}%\n({int(np.round(pct/100.*sum(vals),0))})", pctdistance=0.8)
    for t in outer_text:
        t.update({"color":"black"})
    ax.set(aspect="equal", title="+".join(years))
    ax.legend(inner_wedges, ["Sand", "Loam"])
    plt.tight_layout()
    plt.savefig(f"plots/pie_combined_{prefix}{'_'.join(years)}.png")
    plt.close("all")

def make_pie_charts(data,prefix=""):
    for years in [["2020"], ["2021"], ["2020", "2021"]]:
        year_data = data[data["year"].isin(years)]
        combined_pie_chart(year_data, years, prefix=prefix)


def lin_regression(data):
    g = sns.lmplot(data=data, x="Height", y="Diameter", hue="soiltype", scatter_kws={"alpha":0.3}, truncate=False)
    g.ax.set_title("2020+2021")
    g.ax.set_xlim(50,300)
    g.ax.set_ylim(0,2)
    plt.tight_layout()
    plt.savefig(f"plots/linear_regression.png")
    plt.close("all")


def make_plots(data):
    make_pie_charts(data)
    make_box_plots(data)
    make_p_value_norm_dist_plot(data)
    make_scatter_plots(data)
    lin_regression(data)


def mannwhitneyu_test(data):
    print("# Mann-Whitney-U test #")
    # checking plots
    p_values = []
    for years in [["2020"], ["2021"]]:
        tmp_datay = data[data["year"].isin(years)]
        for s_type in ["sand", "loam"]:
            tmp_datas = tmp_datay[tmp_datay["soiltype"] == s_type]
            for plot in tmp_datas["plot_id"].unique():
                for variable in ["Height", "Diameter", "Height/Diameter"]:
                    u_value, p_value = mannwhitneyu(
                        tmp_datas[tmp_datas["plot_id"] != plot][variable],
                        tmp_datas[tmp_datas["plot_id"] == plot][variable],
                    )
                    p_values.append(p_value)
                    if p_value < 0.05:
                        print(
                            f"p-value for {years}-{plot}-{variable} M-W-U test is below 0.05! ({p_value=:.1e})"
                        )
    ax = sns.histplot(p_values, bins=np.linspace(0, 1, 21))
    ax.set_xlabel("p-value")
    ax.set_ylabel("Count of plot measurements")
    ax.set_title("Mann Whiteny U test")
    plt.vlines([0.05], ymin=0, ymax=8, colors="r", label="0.05")
    plt.savefig("plots/p_values_mann_whitney_dist.png")
    plt.close("all")

    # checking between years
    for s_type in ["sand", "loam"]:
        tmp_data = data[data["soiltype"] == s_type]
        for variable in ["Height", "Diameter", "Height/Diameter"]:
            u_value, p_value = mannwhitneyu(
                tmp_data[tmp_data["year"] == "2020"][variable],
                tmp_data[tmp_data["year"] == "2021"][variable],
            )
            print(
                f"Comparing between 2020 and 2021 - {s_type} - {variable} - Mann Whitney U test {p_value=:.1e}"
            )

    # checking between soiltypes
    for years in [["2020"], ["2021"], ["2020", "2021"]]:
        tmp_datay = data[data["year"].isin(years)]
        for variable in ["Height", "Diameter", "Height/Diameter"]:
            u_value, p_value = mannwhitneyu(
                tmp_datay[tmp_datay["soiltype"] == "loam"][variable],
                tmp_datay[tmp_datay["soiltype"] == "sand"][variable],
            )
            print(
                f"Comparing between sand and loam - {'+'.join(years)} - {variable} - Mann Whitney U test {p_value=:.1e}"
            )


def compile_tex(tex_body):
    with open("plots/tables.tex", "w") as tmp:
        tmp.write("\\documentclass[11pt]{article}\n")
        tmp.write("\\usepackage{booktabs}\n")
        tmp.write("\\begin{document}\n")
        tmp.write(tex_body)
        tmp.write("\\end{document}\n")
    os.system("pdflatex -output-directory=plots plots/tables.tex")
    os.system("rm plots/tables.aux plots/tables.log")

def calc_differences(data):
    variables=["Height", "Diameter", "Height/Diameter"]
    soil_types=["loam", "sand"]
    index=pd.MultiIndex.from_product([variables, soil_types], names=["Variable", "Soil"])
    mi_data=[]
    for variable in variables:
        data20=data[data["year"]=="2020"]
        data21=data[data["year"]=="2021"]
        for s_type in soil_types:
            val20=data20[data20["soiltype"]==s_type][variable]
            val21=data21[data21["soiltype"]==s_type][variable]
            u_value, p_value = mannwhitneyu(val20, val21)
            tmp_data=[f"{val20.median():.1f}", f"{val21.median():.1f}", f"{val21.median()/val20.median()-1:+.1%}", f"{p_value:.1e}"]
            mi_data.append(tmp_data)
    year_comp_df=pd.DataFrame(mi_data, index=index, columns=["2020","2021","Change", "M-W-U p-value"])

    variables=["Height", "Diameter", "Height/Diameter"]
    years_list=[["2020"], ["2021"], ["2020", "2021"]]
    index=pd.MultiIndex.from_product([variables, ["+".join(y) for y in years_list]], names=["Variable", "Years"])
    mi_data=[]
    for variable in variables:
        for years in years_list:
            tmp_datay = data[data["year"].isin(years)]
            val_loam=tmp_datay[tmp_datay["soiltype"]=="loam"][variable]
            val_sand=tmp_datay[tmp_datay["soiltype"]=="sand"][variable]
            u_value, p_value = mannwhitneyu(val_sand, val_loam)
            tmp_data=[f"{val_sand.median():.1f}", f"{val_loam.median():.1f}", f"{val_loam.median()/val_sand.median()-1:+.1%}", f"{p_value:.1e}"]
            mi_data.append(tmp_data)
    soil_comp_df=pd.DataFrame(mi_data, index=index, columns=["Sand","Loam","Difference", "M-W-U p-value"])

    compile_tex(year_comp_df.to_latex() + "\n" + soil_comp_df.to_latex())

def main():
    all_data, precise_data = read_data()
    make_plots(all_data)
    mannwhitneyu_test(all_data)
    calc_differences(all_data)


if __name__ == "__main__":
    main()
