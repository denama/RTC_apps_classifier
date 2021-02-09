#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 12:26:10 2020

@author: dena
"""

bad_domains=set("co.uk co.jp co.hu co.il com.au co.ve .co.in com.ec com.pk co.th co.nz com.br com.sg com.sa \
com.do co.za com.hk com.mx com.ly com.ua com.eg com.pe com.tr co.kr com.ng com.pe com.pk co.th \
com.au com.ph com.my com.tw com.ec com.kw co.in co.id com.com com.vn com.bd com.ar \
com.co com.vn org.uk net.gr".split())


def second_level(fqdn):
    if fqdn[-1] == ".":
        fqdn = fqdn[:-1]
    names = fqdn.split(".")
    if ".".join(names[-2:]) in bad_domains:
        return get3LD(fqdn)
    tln_array = names[-2:]
    tln = ""
    for s in tln_array:
        tln = tln + "." + s
    return tln[1:].lower()


def get3LD(fqdn):
    if fqdn[-1] == ".":
        fqdn = fqdn[:-1]
    names = fqdn.split(".")
    tln_array = names[-3:]
    tln = ""
    for s in tln_array:
        tln = tln + "." + s
    return tln[1:]



def num_to_alpha(list_of_domains):

    alphabetic_list = []
    for domain in list_of_domains:
        for i in range(10):
            domain = domain.replace(str(i), "M")
        alphabetic_list.append(domain)
    return alphabetic_list


bad_words = ["-", "youtube", "office", "twitch", "unito", "studenti", "unito", "football", "overleaf", "dropbox", "lastampa",\
             "repubblica", "adblock", "outlook", "office365", "gmail", "bing", "decathlon", "github", "glovo", "tv", \
             "linkedin", "swas", "gedidigital", "corriere", "grammarly", "drive", "calendar", "fiatgroup", \
             "weather", "fiat", "fcagroup", "chrysler", "harman", "officeapps", "sublimetext"]
#good words = ["jitsi.polito.it"]

def first_filter(list_of_domains):

    filter_func = lambda s: not any(x in s for x in bad_words)
    filtered_list = [line for line in list_of_domains if (filter_func(line)) & (line != "")]

    return filtered_list


def first_filter2(list_of_domains):
    filter_func = lambda s: not any(x in s for x in bad_words)
    filtered_list = [line for line in list_of_domains if (filter_func(line)) & (line != "")]
    filtered_list2 = []
    for domain in filtered_list:
        for i in range(10):
            domain = domain.replace(str(i), "M")
        filtered_list2.append(domain)
    #print("Filtered list", list(filtered_list))
    #print("Filtered list 2", len(filtered_list2))
    #print(len(filtered_list) == len(filtered_list2))

    return filtered_list2


def as_name_to_domain(list_of_as):

    new_list_of_as = []
    for item in list_of_as:
        if type(item) == str:

            if len(item.split()) == 1:
                new_item = ".".join([item, "com"]).lower()

            if len(item.split()) == 2:
                new_item = []
                for word in item.split():
                    new_item.append(word.replace(",", "").replace(".", ""))
                new_item = ".".join(new_item).lower()

            elif len(item.split()) >= 3:
                new_item = []
                for word in item.split()[:2]:
                    new_item.append(word.replace(",", "").replace(".", ""))
                new_item = ".".join(new_item).lower()

            new_list_of_as.append(new_item)

    return new_list_of_as




