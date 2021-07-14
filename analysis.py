import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
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
            tmp_df.drop(
                tmp_df[
                    (tmp_df["Diameter"] == "<0.5")
                    | (tmp_df["Diameter"] == "<0,5")
                    | (tmp_df["Diameter"] == ">0.5")
                    | (tmp_df["Diameter"] == "0")
                    | (tmp_df["Diameter"] == "0,4")
                ].index,
                inplace=True,
            )
            tmp_df["Diameter"] = tmp_df.apply(
                lambda row: float(str(row["Diameter"]).replace(",", ".")), axis=1
            )
            tmp_df["Height"] = tmp_df.apply(
                lambda row: float(str(row["Height"]).replace(",", ".")), axis=1
            )
            tmp_df = tmp_df.astype({"Diameter": float, "Height": float})
            all_data = pd.concat([all_data, tmp_df])
    all_data["Height/Diameter"] = all_data.apply(
        lambda row: row["Height"] / row["Diameter"], axis=1
    )
    all_data.dropna(inplace=True)
    all_data.reset_index(inplace=True, drop=True)
    return all_data


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

def single_pie_chart(data, years, prefix):
    ax=data.value_counts().plot(kind="pie", autopct=lambda val: int(np.round(val/100.*data.value_counts().sum(),0)), ylabel="")
    ax.set_title("+".join(years))
    plt.savefig(f"plots/pie_{prefix}_{'_'.join(years)}.png")
    plt.close("all")

def make_pie_charts(data):
    for years in [["2020"], ["2021"], ["2020", "2021"]]:
        year_data = data[data["year"].isin(years)]
        single_pie_chart(year_data["soiltype"], years, "soiltype_"+"_".join(years))
        for s_type in ["loam", "sand"]:
            single_pie_chart(year_data[year_data["soiltype"]==s_type]["plot_id"], years, f"{s_type}_plot_id_"+"_".join(years))

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


def main():
    willow_data = read_data()
    make_plots(willow_data)
    mannwhitneyu_test(willow_data)


if __name__ == "__main__":
    main()
