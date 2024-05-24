import matplotlib.pyplot as plt
import pandas as pd
from data_cleaning import cleaner
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


df_ice_cream = cleaner()


def production_plot(alias):
    plt.figure(figsize=(10, 4))
    plt.plot(alias.production)
    plt.title('Ice Cream Production over Time', fontsize=20)
    plt.ylabel('Production', fontsize=16)
    for year in range(2011, 2021):
        plt.axvline(pd.to_datetime(str(year)+'-01-01'),
                    color='k', linestyle='--', alpha=0.2)

    plt.show()


def acf_plot(alias):
    plot_acf(alias.production, lags=100)
    plt.show()


def pacf_plot(alias):
    plot_pacf(alias.production, lags=50)
    plt.show()


if __name__ == "__main__":
    production_plot(df_ice_cream)
    # based on decaying ACF, we are likely dealing with an auto regressive process
    acf_plot(df_ice_cream)
    # based on pacf, we should start with an auto regressive model with lags 1,2,3,10,13
    pacf_plot(df_ice_cream)
