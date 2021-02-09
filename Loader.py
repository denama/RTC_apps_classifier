#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 12:04:29 2020

@author: dena
"""

import os
import numpy as np
import pandas as pd

from decode_stacked import decode_stacked

from data_utils import first_filter, second_level, num_to_alpha, as_name_to_domain
from config import config_dict



class Loader ():

    def __init__ (self, conf_dict):
        self.app_list = config_dict["app_list"]
#        self.seconds_before = config_dict["seconds_before"]
#        self.seconds_after = config_dict["seconds_after"]
#        self.use_as = config_dict["use_as"]
        self.seconds_before = conf_dict["seconds_before"]
        self.seconds_after = conf_dict["seconds_after"]
        self.use_as = conf_dict["use_as"]

        self.dff = None
        self.data_pie = None
        self.data_bla = []


        self.dff = self.load_data_new()


    def load_data_new(self):


        data_foler = "Data_enriched"
        data_sources = [
                os.path.join(data_foler, "rtp_tcp_dena_windows_NEW.json"),
                os.path.join(data_foler, "rtp_tcp_new_pcaps_dropbox_NEW.json"),
                os.path.join(data_foler, "rtp_tcp_short_windows_NEW.json"),
                os.path.join(data_foler, "rtp_tcp_windows_blue_NEW.json"),
                os.path.join(data_foler, "rtp_tcp_linux_unuploaded_NEW.json"),
                os.path.join(data_foler, "rtp_tcp_linux_uploaded_NEW.json"),
                os.path.join(data_foler, "rtp_tcp_pcaps_from_dropbox_NEW.json"),
                os.path.join(data_foler, "rtp_tcp_antonio_NEW.json"),
                os.path.join(data_foler, "rtp_tcp_new_additional_NEW.json"),
                ]

        label = []
        self.desc = []
        self.desc_all = []
        pcap_name = []
        second_level_domains = []

        self.all_domains = []
        self.bad_domains = []
        counter = 0
        counter_bad = 0

        for file in data_sources:
            with open(file, "r") as f:
                for obj in decode_stacked(f.read()):

                    if obj["pcap_name"] == "jitsi_dena_thomas_short_no_start.pcapng":
                        continue

                    counter +=1
                    self.data_bla.append(obj)
                    pcap_name.append(obj["pcap_name"])

                    if obj["app"].lower() == "googlemeet":
                        label.append("google_meet")
                    else:
                        label.append(obj["app"].lower())

                    if self.seconds_before == -1:
                        list_to_elaborate = obj["tcp_all"]["c_tls_SNI:116"]
                        description = list(set(num_to_alpha(first_filter(list_to_elaborate))))
                        self.desc.append(description)

#                    elif self.seconds_before == 10:
#                        list_to_elaborate = obj["tcp_before"]["c_tls_SNI:116"]
#                        description = list(set(num_to_alpha(first_filter(list_to_elaborate))))
#                        self.desc.append(description)
                    else:
                        rtp_start = obj["rtp_start"] #in seconds
                        tcp_df = pd.DataFrame(obj["tcp_all"])
                        tcp_after = \
                            tcp_df[((tcp_df.loc[:, "first:29"]/1000) - rtp_start > 0) & \
                                   ((tcp_df.loc[:, "first:29"]/1000 - rtp_start) < self.seconds_after)].copy()
                        domains_after = tcp_after['c_tls_SNI:116'].tolist()
                        as_num_after = tcp_after["ASN"].tolist()
                        #as_name_after = tcp_after["Organization"].dropna().apply(lambda x: x.replace(".", "")).tolist()
                        as_name_after = as_name_to_domain(tcp_after["Organization"].dropna().tolist())

                        tcp_before = \
                            tcp_df[((rtp_start - tcp_df.loc[:, "first:29"]/1000) > 0) & \
                                   ((rtp_start - tcp_df.loc[:, "first:29"]/1000) < self.seconds_before)].copy()
                        domains_before = tcp_before['c_tls_SNI:116'].tolist()
                        as_num_before = tcp_before["ASN"].tolist()
                        #as_name_before = tcp_before["Organization"].dropna().apply(lambda x: x.replace(".", "")).tolist()
                        as_name_before = as_name_to_domain(tcp_before["Organization"].dropna().tolist())

                        if self.use_as:
                            list_to_elaborate = domains_before + domains_after + as_name_before + as_name_after
                        else:
                            list_to_elaborate = domains_before + domains_after

                        description = list(set(num_to_alpha(first_filter(list_to_elaborate))))
                        self.desc.append(description)

                        #asn_list = as_before + as_after


                    second_level_domain = [second_level(domain) for domain in description if "." in domain]
                    second_level_domains.append(second_level_domain)


                    #See bad domains
                    diff = np.setdiff1d(num_to_alpha(list_to_elaborate), description).tolist()
                    for element in diff:
                        self.bad_domains.append(element)

                    #See all domains
                    for domain in list_to_elaborate:
                        if domain:
                            self.all_domains.append(domain)

                    if not description:
                        counter_bad +=1

        self.all_domains = pd.Series(self.all_domains, name="domain")
        self.bad_domains = pd.Series(self.bad_domains).value_counts()

        #print("No TLS domains found in ", counter_bad, "out of " +str(counter)+ " pcaps. \
        #      - Dataset has " +str(len(self.desc))+ " rows.")
        #print(str(len(self.bad_domains)) + " unique domains have been removed by first filter (static list) \
        #      out of " +str(len(self.all_domains))+".")


        self.dff = pd.DataFrame(
                           data = np.array([pcap_name, self.desc, second_level_domains, label], dtype = object).T,
                           columns = ["pcap_name", "description", "second_level_domains", "label"],
                           )

        self.dff["len"] = self.dff["description"].apply(lambda x: len(x))
        self.dff["len_second_level"] = self.dff["second_level_domains"].apply(lambda x: len(x))

        people = ["antonio", "paolo", "gianluca", "michela", "martino",\
                  "maurizio", "francesca", "simone", "rosa", "ico", \
                  "francesco", "daniela", "carlos", "luca", "marko", "dena", "alessandro"]
        self.dff["person"] = self.dff["pcap_name"].apply(lambda x: x.split("_")[0].lower())
        self.dff["person"] = self.dff["person"].apply(lambda x: "dena" if x not in people else x)

        good_names = []
        for name in people:
            if name in self.dff["person"].unique():
                if self.dff["person"].value_counts().loc[name] > 9:
                    good_names.append(name)

        self.dff["person"] = self.dff["person"].apply(lambda x: "other" if x not in good_names else x)

        return self.dff



    def load_data(self):

        data_foler = config_dict["data_folder"]
        data_sources = [
                os.path.join(data_foler, "domains_all_antonio_2020_05_22.json"),
                os.path.join(data_foler, "dena_linux_domains.json"),
                os.path.join(data_foler, "dena_windows_domains.json"),
                os.path.join(data_foler, "dena_small_pcaps_domains.json"),
                os.path.join(data_foler, "domains_dropbox.json"),
                ]


        label = []
        pcap_name = []
        self.desc = []
        second_level_domains = []

        self.bad_domains = []
        self.all_domains = []
        counter = 0
        counter_bad = 0

        #for line in data:
        for path in data_sources:
            with open(path, "r") as f:
                for line in decode_stacked(f.read()):
                    counter +=1
                    if line["tls_domains"][0]:
                        if "" in line["tls_domains"]: line["tls_domains"].remove("")
                        pcap_name.append(line["pcap_name"])
                        label.append(line["app"].lower())

                        description = list(set(num_to_alpha(first_filter(line["tls_domains"]))))
                        self.desc.append(description)

                        second_level_domain = [second_level(domain) for domain in description if "." in domain]
                        second_level_domains.append(second_level_domain)

                        #domains filtered by first filter - bad domains
                        diff = np.setdiff1d(num_to_alpha(line["tls_domains"]), description).tolist()
                        for element in diff:
                            self.bad_domains.append(element)

                        for domain in line["tls_domains"]:
                            if domain:
                                self.all_domains.append(domain)
                    else:
                        #Pcaps that dont have TLS names
                        counter_bad+=1


        self.all_domains = pd.Series(self.all_domains, name="domain")
        self.bad_domains = pd.Series(self.bad_domains).value_counts()
        #pd.DataFrame(all_domains.value_counts()[all_domains.value_counts() > 3]).to_csv("domains.csv")

        print("No TLS domains found in ", counter_bad, "out of " +str(counter)+ " pcaps. \
              - Dataset has " +str(len(self.desc))+ " rows.")
        print(str(len(self.bad_domains)) + " unique domains have been removed by first filter (static list) \
              out of " +str(len(self.all_domains))+".")

        self.dff = pd.DataFrame(data = np.array([pcap_name, self.desc, second_level_domains, label], dtype = object).T, \
                           columns = ["pcap_name", "description", "second_level_domains", "label"],
                           #dtype = object,
                           )
        self.dff["len"] = self.dff["description"].apply(lambda x: len(x))
        self.dff["len_second_level"] = self.dff["second_level_domains"].apply(lambda x: len(x))

        people = ["antonio", "paolo", "gianluca", "michela", "martino",\
                  "maurizio", "francesca", "simone", "rosa", "ico", \
                  "francesco", "daniela", "carlos", "luca", "marko", "dena", "alessandro"]
        self.dff["person"] = self.dff["pcap_name"].apply(lambda x: x.split("_")[0].lower())
        self.dff["person"] = self.dff["person"].apply(lambda x: "dena" if x not in people else x)


        good_names = []
        for name in people:
            if name in self.dff["person"].unique():
                if self.dff["person"].value_counts().loc[name] > 9:
                    good_names.append(name)

        self.dff["person"] = self.dff["person"].apply(lambda x: "other" if x not in good_names else x)

        return self.dff



    def get_per_app_domains_count(self):

        d = {}
        for name in self.dff["label"].unique():
            d[name] = self.dff[["description", "label"]][self.dff["label"] == name].copy()
        #print(d)

        #domains has key: app and value: Series of all domains found for that app
        self.domains = {k: [] for k in self.dff["label"].unique()}
        for key, value in d.items():
            series = value["description"]
            for l in series:
                for i in l:
                    if i: self.domains[key].append(i)
            self.domains[key] = pd.Series(self.domains[key]).value_counts() \
                                .rename("occurences").reset_index() \
                                .rename(columns={"index": "domain"}
                                )

        total_per_domain = self.dff["label"].value_counts()

        for app, value in self.domains.items():
            value.insert(value.shape[1], "occurences_perc", value["occurences"]/total_per_domain[app])

        return self.domains


    def plot_pie_apps(self, which_apps="to_classify", title_addon=""):

        import plotly.graph_objects as go

        save_folder = "Plots_data"

        if which_apps == "to_classify":
            self.data_pie = self.dff[self.dff["label"].isin(self.app_list)]["label"].value_counts()
            title = "Number of samples of each class "+title_addon
        else:
            self.data_pie = self.dff["label"].value_counts()
            title = "Number of samples of each app in dataset "+ title_addon


#def plot_pie(values, labels, title, save_folder):
        fig = go.Figure(data=[go.Pie(labels=self.data_pie.index, values=self.data_pie, hole=.3)])

        fig.update_traces(hoverinfo='label+percent', textinfo='label+value', textfont_size=20,
                          #marker=dict(colors=colors, line=dict(color='#000000', width=2))
                         )
        fig.update_layout(
                        legend = dict(font=dict(size=28), x=1),
                        title = dict(text=title, font={"size": 32}),
                        autosize=True,
                        #autosize=False,
                        #width=1500,
                        #height=800,
        )
        fig.show()
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        fig.write_html(os.path.join(save_folder, title+".html"))
        fig.write_image(os.path.join(save_folder ,title+".png"))




