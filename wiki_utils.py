#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:45:15 2021
Rev 2022-02-16
Rev 2022-05-30
Rev 2022-06-20
Rev 2022-07-05
Rev 2022-09-12
Rev 2023-05-25
Rev 2023-10-06
Rev 2023-11-18
Rev 2024-05-19 vers. 1.2.5

@author: Angel Zazo <angelzazo@usal.es>
"""

#%% Imports

import sys
from time import time, sleep
import requests
import pandas as pd
from http.cookiejar import http2time
import regex as re
import numpy as np
from io import StringIO
from collections import Counter
from os.path import splitext   # split path in name and extension
import unicodedata
from difflib import SequenceMatcher


#%% GLOBAL VARIABLES
# General user_agent header for Wikimedia, Wikidata, MediaWiki and VIAF requests
# See https://www.mediawiki.org/wiki/API:Etiquette
# See https://meta.wikimedia.org/wiki/User-Agent_policy.
user_agent = "wikiTools Package - Python/%s.%s" % (sys.version_info[0], sys.version_info[1])

# In MediaWiki API specifying titles through titles or pageids in the query API
# is limited to 50 titles per query, or 500 for those with the "apihighlimits" right.
# See https://www.mediawiki.org/wiki/API:Query#Additional_notes.
MW_LIMIT = 50

# See https://www.oclc.org/developer/api/oclc-apis/viaf/authority-cluster.en.html
# VIAF API restriction is 250 maximun returned records.
VIAF_LIMIT = 250


#%% doChunks(f, x, chunksize, **arg)
def doChunks(f, x, chunksize, **arg):
  """
  Execute the function f(x,chunksize,...) in chunks of chunksize elements each.

  The Wikidata and Wikimedia APIs impose limits on query execution. Wikidata,
  for instance, has a timeout limit of 60 seconds, meaning that larger queries
  involving more entities increase the risk of hitting this limit. Similarly,
  Wikimedia APIs enforce restrictions on the number of titles or page IDs that
  can be included in a single query. To prevent errors arising from these
  limitations, this function executes the function 'f' sequentially over
  chunks of elements.

  :param f: The function to execute. The function is expected to return a
            Pandas dataframe.
  :param x: List of Wikidata entities or titles/pageids of Wikimedia pages.
  :param chunksize: Maximum number of elements in `x` used in each execution.
  :param arg: Parameters to be pased to the function. It is mandatory to
              use named parameters (like a dict).
  :return Concatenation of Pandas Dataframes returned by 'f' on each chunk of x.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  """
  debug = 'debug' in arg and arg['debug'] == 'info'
  if debug:
    timeinit = time()
  n = len(x)
  nlim = int(n/chunksize)
  for k in range(nlim+1):
    offset = k*chunksize
    x_list = x[offset:offset+chunksize]
    if len(x_list)==0:
      break
    if debug:
      t0=time()
      print(f" INFO: Executing the function on elements from {offset+1} to {offset+len(x_list)}", end="", file=sys.stderr)
    d = f(x_list, chunksize=chunksize, **arg)
    if debug:
      print(f" ({time()-t0:.2f} seconds)", file=sys.stderr)
    if d is None:
      return None
    if (k==0):
      output = d
    else:
      if isinstance(d,dict):
        output.update(d)
      elif isinstance(d,tuple):
        d0 = d[0]  # d[0] can be dict or Pandas-Datafreme,
        d1 = d[1]  # d[1] always is a dict
        output[1].update(d1)
        if isinstance(d0,dict):
          output[0].update(d0)
        else:
          output = (pd.concat([output[0], d0]), output[1])
      else:
        output = pd.concat([output, d])
  if debug:
    print(f" INFO: Total time {time()-timeinit:.2f} seconds", file=sys.stderr)
  return(output)


#%% checkEntities(entity_list)
def checkEntities(entity_list):
  """
  Check if all Wikidata entities in entily_list have valid values. Return a
  list of entities with duplicates removed or raise error.

  :param entity_list: A Wikidata entity (Qxxx, Pxxx) or a list of entities.
  :return A list of entities or raises ValueError exception.
  """
  if isinstance(entity_list, str):
    entity_list = [entity_list]
  entity_list = [x.strip() for x in entity_list if not re.match('\s*$', x)]
  if len(entity_list) == 0:
    raise ValueError("Invalid value for parameter 'entity_list'")
  for q in entity_list:
    m = re.match(r'(?:Q|P)\d+$', q)
    if m is None:
      raise ValueError(f"Invalid value in parameter 'entity_list': '{q}'")
  # Remove duplicates preserving order
  entity_list = list(dict.fromkeys(entity_list))
  return entity_list


#%% -- WDQS: WikiData Query Service -------------------------------------------
#  Use the WQS SPARQL endpoint.
# See https://query.wikidata.org/
# See https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial
# See https://www.mediawiki.org/wiki/Wikidata_Query_Service/User_Manual

#%% def reqWDQS(sparql_query,  method='GET', format='json'):
def reqWDQS(sparql_query, method='GET', format='json'):
  """
  Make a request to Wikidata Query Service (WDQS) SPARQL endpoint.

  :param sparql_query: The query in SPARQL language (a SELECT query).
  :param method: The method used to send the request, GET or POST, mandatory.
         Default 'GET'. Use 'POST' method for long SELECT clauses.
  :param format: The response format: only 'json', 'xml' or 'csv' formats are
         allowed, default 'json'. If format='csv' the function returns a
         Pandas dataframe in which column names are the variable names of the
         SELECT query (note that in this case, all data are strings.)
         See https://www.mediawiki.org/wiki/Wikidata_Query_Service/User_Manual#SPARQL_endpoint
         and https://www.wikidata.org/wiki/Wikidata:Data_access/es

  :return The response in the format selected.
  :raise Exception: From response.raise_for_status() or other exception.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  """
  if format=='json':
    wdqs_format = "application/sparql-results+json"
  elif format=='xml':
    wdqs_format = "application/sparql-results+xml"
  elif format=='csv':
    wdqs_format = "text/csv"
  else:
    raise ValueError(f"Format '{format}' is not supported")
  #
  url = 'https://query.wikidata.org/sparql'
  params = {'query': sparql_query}
  headers = {'user-agent': user_agent,
             'accept': wdqs_format,
             'Accept-Encoding': "gzip, deflate"  # default in requests package
             }
  # Exit from while in return or exception in response.raise_for_status
  while True:
    if method=='GET':
      response = requests.get(url=url, params=params, headers=headers)
    elif method=='POST':
      response = requests.post(url=url, data=params, headers=headers)
    else:
      raise ValueError(f"Method '{method}' is not supported")
    # See https://www.mediawiki.org/wiki/Wikidata_Query_Service/User_Manual#Query_limits
    if response.status_code == 429:
      t = response.headers['Retry-after']
      m = re.match(r'^\d+$', t)
      if m is None:
        t = int(http2time(t) - int(time()))
      else:
        t = int(t)
      if t > 600:
        raise("ERROR: receive a 429 status-code response, but retry-after > 600")
      print(f"Received a 429 status-code response. Sleeping {t} seconds",
            file=sys.stderr)
      sleep(t)
      continue
    #
    response.raise_for_status()
    rtype = response.headers['Content-Type']
    if format == 'json' and rtype.startswith("application/sparql-results+json"):
      return response.json()
    if format == 'xml' and rtype.startswith("application/sparql-results+xml"):
      return response.text
    if format == 'csv' and rtype.startswith("text/csv"):
      return pd.read_csv(StringIO(response.text, newline=''), dtype=str)
    else:
      raise ValueError(f"reqWDQS() format '{format}' or response type '{rtype}' is incorrect")

#%% w_isInstanceOf(entity_list, instanceof='', chunksize=50000, debug=False)
def w_isInstanceOf(entity_list, instanceof='', chunksize=50000, debug=False):
  """
  Check using WDQS if the Wikidata entities in 'entity_list' are instances of
  'instanceof' Wikidata entity class. For example, if instanceof="Q5", checks
  if entities are instances of the Wikidata entity class Q5, i.e, are humans.
  Some entity classes are allowed, separated by '|', in this case, the OR
  operator is considered. If instanceOf='' then no filter is applied: the
  funtion returns the Wikidata entities class of which each of the entities
  in the list are instances. Duplicated entitites in the 'entity_list'
  are deleted before search.
  Uses `doChunks` function if the number of entities is greather that the value
  of 'chunksize' parameter.
  Note that no labels or descriptions of the entities are returned. Please, use
  the `w_LabelDesc` function for this.

  :param entity_list: Wikidata entity or a list of Wikidata entities.
  :param instanceof: The Wikidata class to check. Some entity
         classes are allowed, separated by '|', in this case, the OR
         operator is considered.
  :param chunksize: If the number of entities exceeds this number, several
         requests will be made. This is the maximum number of entities
         requested in each request. Please, decrease the default value if error
         is raised.
  :param debug: Three values are allowed: False (no debugging information is
         shown), 'info' (only information about chunked queries), 'query' (in
         addition to the information shown by 'info', the query launched to the
         API is also showed).
  :return A data-frame with three columns, first Wikidata entity, second all
         Wikidata class each instance is instance of them, last TRUE or FALSE
         if each entity is instance of the `instanceof` parameter, if this one
         is set.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  Note: It is easy to know which entities are not instances of "instaceof":
  >>> d = w_isInstanceOf(['Q9022', 'Q557', 'Q873'], instanceof = 'Q5')
  >>> d
  >>> d.columns
  Index(['entity', 'instanceof_Q5'], dtype='object')
  >>> d[~ d.instanceof_Q5]
        entity  instanceof_Q5
  Q9022  Q9022          False
  """
  # Checking entity_list
  entity_list = checkEntities(entity_list)
  #
  n = len(entity_list)
  # Number of entities exceeds chunksize:
  if n > chunksize:
    if debug:
      print(f"INFO: The number of entities ({n}) exceeds chunksize ({chunksize}).", sep="", file=sys.stderr)
    return doChunks(w_isInstanceOf, entity_list, chunksize,
                    instanceof=instanceof, debug=debug)
  #
  values     = "wd:" + " wd:".join(entity_list)
  #
  query = f"""SELECT ?entity
(GROUP_CONCAT(DISTINCT ?instanc;separator="|") as ?instanceof)
WHERE {{
  OPTIONAL {{
    VALUES ?entity {{ {values} }}
    OPTIONAL {{?entity wdt:P31 ?instanc.}}
  }}
}} GROUP BY ?entity"""
  #
  if debug=='query':
    print(query, file=sys.stderr)
  #
  d = reqWDQS(query, method='POST', format="csv")
  d.entity = d.entity.str.slice(31)
  d.instanceof = d.instanceof.replace('http://www.wikidata.org/entity/', '', regex=True)
  #
  # If instanceof!='' => set new true/false column
  if instanceof!='':
    d['instanceof_' + instanceof] = d.instanceof.str.contains(r'\b(?:' + instanceof + r')\b', regex=True)
  #
  d.index = d.entity
  return d


#%% w_Wikipedias(entity_list, wikilangs="", instanceof='', chunksize=10000, debug=False)
def w_Wikipedias(entity_list, wikilangs="", instanceof='', chunksize=10000, debug=False):
  """
  Get Wikipedia page titles and URLs of the Wikidata entities in entity_list.

  Get from Wikidata all Wikipedia page titles and URL of the Wikidata entities
  in entity_list. If parameter `wikilangs`='', then returns all Wikipedia page
  titles, else only the languages in `wikilangs`. The returned dataframe also
  includes the Wikidata entity classes of which the searched entity is
  an instance. If parameter `instanceof`!='', then only returns the pages
  for Wikidata entities which are instances of the Wikidata class indicated in
  it. The data-frame doesn't return labels or descriptions about entities: the
  function `w_LabelDesc` can be used for this. Duplicated entities are deleted
  before search. Index of the data-frame returned are also set to entity_list.

  Uses `doChunks` function if the number of entities is greather that the value
  of 'chunksize' parameter.

  :param entity_list: Wikidata entity or a list of Wikidata entities.
  :param wikilangs: List of languages to limit the search, using "|" as
         separator. Wikipedias page titles are returned in same order as
         languages in this parameter. If wikilangs='' the function returns
         Wikipedia page titles in any language, not sorted.
  :param instanceof: Wikidata entity class to limit the result to the instances
         of that class. For example, if instanceof='Q5', limit the results to
         "human".
  :param chunksize: If the number of entities exceeds this number, several
         requests will be made. This is the maximum number of entities
         requested in each request.
         Default value (1500) is a good choice. If parameter wikilangs!="",
         this limit can be increased.
  :param debug: For debugging purposes (default FALSE). If debug='info'
         information about chunked queries is shown. If debug='query' also the
         query launched is shown.
  :return A Pandas data-frame with five columns: entities, instanceof, npages,
          page titles and page URLs. Last three use "|" as separator. Index of
          the data-frame is also set to the entity_list. Returns None on errors.
  :example
    >>> # aux: get a vector of entities (l).
    >>> df =  w_SearchByLabel(string='Napoleon', langsorder='en', mode='inlabel')
    >>> l = df.entity  # approx. 3600
    >>> # Run function with differents parameter values
    >>> w = w_Wikipedias(entity_list=l, debug='info')
    >>> w = w_Wikipedias(entity_list=l, wikilangs='es|en|fr', debug='info')
    >>> # Filter if instanceof=Q5 (human):
    >>> w_Q5 = w_Wikipedias(entity_list=l, wikilangs='es|en|fr', instanceof='Q5', debug='info')
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  """
  #
  # Check entity_list
  entity_list = checkEntities(entity_list)
  #
  n = len(entity_list)
  # Number of entities exceeds chunksize:
  if n > chunksize:
    if debug!=False:
      print(f"INFO: The number of entities ({n}) exceeds chunksize ({chunksize}).", sep="", file=sys.stderr)
    return doChunks(w_Wikipedias, entity_list, chunksize, wikilangs=wikilangs,
                    instanceof=instanceof, debug=debug)
  #
  values = "wd:" + " wd:".join(entity_list)
  #
  if wikilangs=="":
    w_filter = ""
  else:
    wikiorder = wikilangs.split('|')
    w_filter = "FILTER(?lang IN ('" + "', '".join(wikiorder) + "'))"
  #
  # Note the OPTIONAL before VALUES, otherwise we get timeouts.
  query = f"""SELECT DISTINCT ?entity
(GROUP_CONCAT(DISTINCT ?instanc;separator="|") as ?instanceof)
(COUNT(DISTINCT ?page) as ?npages)
(GROUP_CONCAT(DISTINCT ?lang;separator="|") as ?langs)
(GROUP_CONCAT(DISTINCT ?name;separator="|") as ?names)
(GROUP_CONCAT(DISTINCT ?page;separator="|") as ?pages)
WHERE {{
  OPTIONAL {{
    VALUES ?entity {{ {values} }}
    OPTIONAL {{?entity wdt:P31 ?instanc.}}
    OPTIONAL {{
    ?page schema:about ?entity;
          schema:inLanguage ?lang;
          schema:name ?name;
          schema:isPartOf [wikibase:wikiGroup "wikipedia"].
          {w_filter}
    }}
  }}
}} GROUP BY ?entity
"""
  #
  if debug=='query':
    print(query, file=sys.stderr)
  #
  d = reqWDQS(query, method='POST', format='csv')
  #
  d.fillna('', inplace=True)
  # Remove http://www.wikidata.org/entity/" [32 chars]
  d.entity = d.entity.str.slice(31)
  d.instanceof = d.instanceof.str.replace('http://www.wikidata.org/entity/', '', regex=False)
  d.npages = d.npages.astype(np.int16)
  d.index = d.entity.values
  # Filtering instanceof
  if instanceof!='':
    d = d[d.instanceof.str.contains(r'\b' + instanceof + r'\b')]
  # Set order of langs, names and pages
  if wikilangs!='':
    for qid in d.index:
      if d.at[qid,'npages']>1:  # Sorting only if number of wikipedia pages > 1
        l = d.at[qid,'langs'].split('|')
        order = [l.index(x) for x in wikiorder if x in l]
        for k in ['langs', 'names', 'pages']:
          r = d.at[qid,k].split('|')
          d.at[qid,k] = '|'.join([r[i] for i in order])
  #
  return d

#%% w_isValid(entity_list, chunksize=50000, debug=False)
def w_isValid(entity_list, chunksize=50000, debug=False):
  """
  Check if the Wikidata entities in 'entity_list' are valid: an entity is valid
  if it has a label or has a description in Wikidata. If one entity exists but
  is not valid, is possible that it redirects to another entity, in that
  case, the target entity is obtained. Other entities may have existed in the
  past, but have been deleted.
  Duplicated entities in entity_list are deleted before checking.
  This function returns a Panda data-frame. The index of the data-frame are
  also set to entity_list. The data-frame also includes the Wikidata entity
  classes which are instance of the entities searched.

  Uses `doChunks` function if the number of entities is greather that the value
  of 'chunksize' parameter.

  Note that no labels or descriptions of the entities are returned. Please, use
  the w_LabelDesc function for this.

  :param entity_list: Wikidata entity or a list of Wikidata entities.
  :param chunksize: If the number of entities exceeds this number, several
         requests will be made. This is the maximum number of entities
         requested in each request. Please decrease the default value if an
         error occurs.
  :return A data-frame with four columns: first, the entity itself, second, if
          that entity is valid in Wikidata (TRUE or FALSE), third, the
          corresponding Wikidata class of which the entities are instances (if
          entities are valid, of course), last, if the entity redirects to
          another Wikidata entity, this entity.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :example
  >>> d = w_isValid(["Q9021", "Q115637688", "Q105660123"])
  >>> d
                  entity  valid instanceof     target
  Q9021            Q9021   True         Q5
  Q105660123  Q105660123  False             Q97352588
  Q115637688  Q115637688  False
  >>> sum(d.valid)
  1
  >>> # Large list
  ... l = w_SearchByOccupation(Qoc='Q2306091') #  Sociologist
  ... l = list(l.entity)
  ... l.extend(["Q115637688", "Q105660123"])  # Adding two new entities
  >>> v = w_isValid(l, debug='info')
  >>> # Not valid
  ... v[~ v.valid]
                  entity  valid instanceof     target
  Q105660123  Q105660123  False             Q97352588
  Q115637688  Q115637688  False
  """
  # Checking entity_list
  entity_list = checkEntities(entity_list)
  #
  n = len(entity_list)
  # Number of entities exceeds chunksize:
  if n > chunksize:
    if debug:
      print(f"INFO: The number of entities ({n}) exceeds chunksize ({chunksize}).", sep="", file=sys.stderr)
    return doChunks(w_isValid, entity_list, chunksize, debug=debug)
  #
  values = "wd:" + " wd:".join(entity_list)
  #
  query = """SELECT ?entity ?valid
(GROUP_CONCAT(DISTINCT ?instanc; separator='|') as ?instanceof) ?redirection
WHERE {
  OPTIONAL {
    VALUES ?entity {""" + values + """ }
    BIND(EXISTS{?entity rdfs:label []} || EXISTS{?entity schema:description []} AS ?valid).
    OPTIONAL {?entity wdt:P31 ?instanc.}
    OPTIONAL {?entity owl:sameAs ?redirection}
  }
} GROUP BY ?entity ?valid ?redirection
"""
  #
  if debug=='query':
    print(query, file=sys.stderr)
  #
  d = reqWDQS(query, method='POST', format="csv")
  #
  d.entity      = d.entity.str.slice(31)
  d.redirection = d.redirection.str.slice(31)
  d.instanceof  = d.instanceof.replace(to_replace='http://www.wikidata.org/entity/', value='', regex=True)
  d.valid = d.valid.apply(lambda x: True if x=='true' else False)
  d.index = d.entity.values
  d.fillna('', inplace=True)
  return d


#%% w_Property(entity_list, Pproperty, includeQ=FALSE, langsorder='en',
#               chunksize=5000, debug=False)
def w_Property(entity_list, Pproperty, includeQ=False, langsorder='en',
                chunksize=5000, debug=False):
  """
  Get the properties indicated in the parameter 'Pproperty' for the
  entities in 'entity_list' using the language order in 'langsorder'. The
  function also returns the Wikidata class of which the entities are instances
  of. If parameter `includeQ` is True, also is returned the Wikidata entities
  for the properties. Duplicates in entity_list are deleted before search.
  Information about properties is returned in the order of languages in
  'langsorder'.

  Uses `doChunks` function if the number of entities is greather that the value
  of 'chunksize' parameter.

  Note that no labels or descriptions of the entities are returned. Please, use
  the w_LabelDesc function for this.

  :param entity_list: Wikidata entity or a list of Wikidata entities.
  :param Pproperty: Wikidata properties to search, separated with '|',
         mandatory. For example, is Pproperty="P21", the results contain
         information of the sex of entities. If Pproperty="P21|P569" also shows
         the birthdate (P569). If Pproperty='P21|P569|P214' also shows the VIAF
         identifier (P214). Mandatory.
  :param includeQ If the value is TRUE the function returns the Wikidata entity
         (Qxxx) of the Pproperty. If also `langsorder` has language(s), the
         labels, if any, are returned too. Note that includeQ is only effective
         if `Pproperty` corresponds with a Wikidata entity, else the same
         values that label are returned.
  :param langsorder: Order of languages in which the information will be
         returned, separated with '|'. If no information is given in the first
         language, next is used. This parameter is mandatory if parameter
         `includeQ` is False. If includeQ=True and langsorder='' no labels are
         returned.
  :param chunksize: If the number of entities exceeds this number, several
         requests will be made. This is the maximum number of entities
         requested in each request. Please decrease the default value if an
         error occurs.
  :return A data-frame with the entity, the entities of the properties and the
          labels in langsorder for them.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :example
  >>> d = w_Property(["Q1252859", "Q712609", "Q381800"], Pproperty='P21|P569|P214',
                      langsorder='en|es')
  >>> d
              entity instanceof   P21                  P569      P214
  Q381800    Q381800         Q5  male  1862-12-07T00:00:00Z   4938246
  Q712609    Q712609         Q5  male  1908-05-23T00:00:00Z  36092166
  Q1252859  Q1252859         Q5  male  1913-11-02T00:00:00Z  40787112
  >>> # Large list#'
  ... df = w_SearchByOccupation(Qoc='Q2306091') # ~ 20000
  ... l = df$entity
  >>> p = w_Property(l, Pproperty='P21|P569|P214', langsorder='es|en', debug='info')
  >>> p = w_Property(l, Pproperty='P19', langsorder='', includeQ=True, debug='info')

"""
  # Checking entity_list
  entity_list = checkEntities(entity_list)
  #
  Pproperty = Pproperty.strip()
  langsorder = langsorder.strip()
  if Pproperty=='' or (langsorder=='' and not includeQ):
    raise ValueError("ERROR: one or more parameters 'Pproperty', 'includeQ' or 'langsorder' are incorrect.")
  #
  n = len(entity_list)
  # Number of entities exceeds chunksize:
  if n > chunksize:
    if debug:
      print(f"INFO: The number of entities ({n}) exceeds chunksize ({chunksize}).", sep="", file=sys.stderr)
    return doChunks(w_Property, entity_list, chunksize, Pproperty=Pproperty,
                    includeQ=includeQ, langsorder=langsorder, debug=debug)
  #
  values = "wd:" + " wd:".join(entity_list)
  searchlang = ''
  #
  group_concat = []
  search = []
  searchlang = []
  if langsorder!='':
    langsorder = langsorder.replace("|", ",")
    searchlang.append(f'\n SERVICE wikibase:label {{bd:serviceParam wikibase:language "{langsorder}".\n  ?instanc rdfs:label ?instancLabel.')
    group_concat.append("(GROUP_CONCAT(DISTINCT ?instancLabel; separator='|') as ?instanceofLabel)")
  #
  pprops = Pproperty.split('|')
  for p in pprops:
    if includeQ:
      group_concat.append(f"(GROUP_CONCAT(DISTINCT ?{p}p;separator='|') as ?{p})")
    if langsorder!='':
      group_concat.append(f"(GROUP_CONCAT(DISTINCT STR(?{p}label);separator='|') as ?{p}Label)")
      searchlang.append(f'  ?{p}p rdfs:label ?{p}label.')
    search.append(f'    OPTIONAL {{?entity wdt:{p} ?{p}p.}}')

  group_concat = '\n'.join(group_concat)
  search       = '\n'.join(search)
  searchlang   = '\n'.join(searchlang)
  searchlang += '}' if searchlang!='' else ''
  #
  query = f"""SELECT ?entity
(GROUP_CONCAT(DISTINCT ?instanc; separator='|') as ?instanceof)
{group_concat}
WHERE {{
  OPTIONAL {{
    VALUES ?entity {{ {values} }}
    OPTIONAL {{?entity wdt:P31 ?instanc.}}
{search}
{searchlang}
  }}
}} GROUP BY ?entity"""
  #
  if debug=='query':
    print(query, file=sys.stderr)
  #
  d = reqWDQS(query, method='POST', format="csv")
  #
  d.entity = d.entity.str.slice(31)
  if includeQ:
    for p in pprops:
      d[p] = d[p].replace(to_replace='http://www.wikidata.org/entity/', value='', regex=True)
  d.instanceof = d.instanceof.replace(to_replace='http://www.wikidata.org/entity/', value='', regex=True)
  d.index = d.entity.values
  d.fillna('', inplace=True)
  return d


#%% w_Geoloc(entity_list, langsorder='', chunksize=1000, debug=False)
def w_Geoloc(entity_list, langsorder='', chunksize=1000, debug=False):
  """
  Get Latitude and Longitude coordinates, and country of the Wikidata entities
  in entity_list if any. If 'langsorder'='', then no labels or descriptions
  of the entities are returned, otherwise the function returns them in the
  language order indicated in this parameter. Index of the data-frame is also
  set to entity_list.

  Uses `doChunks` function if the number of entities is greather that the value
  of 'chunksize' parameter.

  Note that Sometimes an old place has several actualplaces, so, it's necessary
  select only one (for example, place=Q18097 [Korea] are two replacement
  places, North Korea and South Korea, so, the Pandas dataframe returned has
  duplicated index). Other times the place is a border between two countries.
  There are some approaches (number of references, timestamps, etc.) to solve
  this problem, but they are not consistent. In general, this happens very
  seldom, so dropped duplicates can be a good option.

  :param entity_list: Wikidata entity or a list of Wikidata entities.
  :param langsorder: Order of languages in which the information will be
         returned, separated with '|'. If no information is given in the first
         language, next is used. If langsorder=='', then labels or descriptions
         are not returned.
  :param chunksize: If the number of entities in the database or authorities'
         catalog exceeds this number, then query are made in chunks. Please,
         decrease the default value if an error occurs.
  :return A data-frame with 'entity', label, Latitude and Longitude, country
         and label of the country.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :example
  >>> d = w_Geoloc(["Q57860", "Q90", "Q15695"], langsorder="")
  >>> d
          placeQ           lat           lon countryQ
  Q15695  Q15695        40.965  -5.664166666      Q29
  Q90        Q90  48.856666666   2.352222222     Q142
  Q57860  Q57860  59.898888888  10.964166666      Q20
  >>> d = w_Geoloc(["Q57860", "Q90", "Q15695"], langsorder="es")
  >>> d
          placeQ      place           lat           lon countryQ  country
  Q57860  Q57860  Lørenskog  59.898888888  10.964166666      Q20  Noruega
  Q15695  Q15695  Salamanca        40.965  -5.664166666      Q29   España
  Q90        Q90      París  48.856666666   2.352222222     Q142  Francia
  >>> # Large list
  ... df = w_SearchByOccupation(Qoc='Q2306091') # ~ 20000
  ... l = df.entity
  ... # Get birth-place (P19)
  ... p = w_Property(l, Pproperty = 'P19', includeQ=True, langsorder='', debug='info')
  ... # Filter entities that have places
  ... places = p.P19[p.P19.str.contains(r"^Q\d+$")].unique() # ~ 4000
  >>> g = w_Geoloc(places, langsorder='en|es', debug='info')

  """
  # Remove duplicates preserving order
  entity_list = checkEntities(entity_list)
  #
  n = len(entity_list)
  # Number of entities exceeds chunksize:
  if n > chunksize:
    if debug:
      print(f"INFO: The number of entities ({n}) exceeds chunksize ({chunksize}).", sep="", file=sys.stderr)
    return doChunks(w_Geoloc, entity_list, chunksize, langsorder=langsorder,
                    debug=debug)
  #
  values = "wd:" + " wd:".join(entity_list)
  #
  langsorder = langsorder.strip()
  if langsorder == '':
    sss = '?place (STR(SAMPLE(?clat)) as ?placeLat) (STR(SAMPLE(?clon)) as ?placeLon) ?country'
    sqq = ''
    sgg = '?place ?country'
  else:
    sss = '?place ?placeLabel (STR(SAMPLE(?clat)) as ?placeLat) (STR(SAMPLE(?clon)) as ?placeLon) ?country ?countryLabel'
    langsorder = langsorder.replace('|', ',')
    sqq = f"""SERVICE wikibase:label {{bd:serviceParam wikibase:language "{langsorder}".
    ?place rdfs:label ?placeLabel.\n    ?country rdfs:label ?countryLabel.}}"""
    sgg = '?place ?placeLabel ?country ?countryLabel'
  #
  query = f"""SELECT DISTINCT {sss}
WHERE {{
  OPTIONAL {{
    VALUES ?place {{ {values} }}
    OPTIONAL {{?place wdt:P1366+ ?placelast.}}
    OPTIONAL {{?place wdt:P625 ?c1.}}
    OPTIONAL {{?placelast wdt:625 ?c2.}}
    BIND(COALESCE(?c1, ?c2) AS ?c).
    BIND(geof:longitude(?c) AS ?clon)
    BIND(geof:latitude(?c)  AS ?clat)
    BIND(COALESCE(?placelast, ?place) AS ?actualplace).
    OPTIONAL {{
      ?actualplace wdt:P17 ?country.
      ?country wdt:P31 ?instance.
      FILTER (?instance in (wd:Q3624078, wd:Q7275, wd:Q6256)).
      #not a former country
      # FILTER NOT EXISTS {{?countryQ wdt:P31 wd:Q3024240}}
      #and no an ancient civilisation (needed to exclude ancient Egypt)
      # FILTER NOT EXISTS {{?countryQ wdt:P31 wd:Q28171280}}
      FILTER (?instance not in (wd:Q3024240, wd:Q28171280)).
    }}
  }}
  {sqq}
}} GROUP BY {sgg}"""

  if debug=='query':
    print(query, file=sys.stderr)
  #
  d = reqWDQS(query, method="POST", format="csv")
  #
  d.place   = d.place.str.slice(31)
  d.country = d.country.str.slice(31)
  d.index = d.place.values
  #
  # Drop duplicated (see docstring)
  d = d[~d.index.duplicated(keep='first')]
  d.fillna('', inplace=True)
  #
  return d

#%% w_LabelDesc(entity_list, what='LD', langsorder='en', chunksize=25000,
#               debug=False)
def w_LabelDesc(entity_list, what='LD', langsorder='en', chunksize=25000, debug=False):
  """
  Return label and/or descriptions of the entities in entity_list in language
  indicated in langsorder. Note that entities can be Wikidata entities (Qxxx)
  or Wikidata properties (Pxxx).

  Use WDQS API. The Wikibase API has limit to 50 entities per query, so is very
  slow to obtains the information. It is also limited to a single language and
  has no language failback.

  Uses `doChunks` function if the number of entities is greather that the value
  of 'chunksize' parameter.

  :param entity_list: Wikidata entity or a list of Wikidata entities. Note that
         duplicated entities are delete before search.
  :param what: Retrieve only Labels (L), only Descriptions (D) or both (LD).
  :param langsorder: Order of languages in which the information will be
         returned, separated with '|'. If no information is given in the first
         language, next is used. This parameter is mandatory, at least one
         language is required, default 'en'.
  :param chunksize: If the number of entities exceeds this number, several
         requests will be made. This is the maximum number of entities
         requested in each request. Please decrease the default value if an
         error occurs.
  :param debug: If True the query launched to the WDQS is shown.
  :return A Pandas data-frame with one column for the entities, and others for
          the language and the labels and/or descriptions. The index of the
          dataframe is also set to the entity list.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :example
  >>> d = w_LabelDesc(["Q57860", "Q712609", "Q381800", "P569"], what='LD',
                      langsorder='se|es|en')
  >>> d
              entity langLabel           label langsdesc         description
  Q1252859  Q1252859        en   Rafael Aburto        en   Spanish architect
  Q712609    Q712609        en  Max Abramovitz        en  American architect
  Q381800    Q381800        en       Paul Adam        en     French novelist
  """
  # Checking entity_list
  entity_list = checkEntities(entity_list)
  #
  langsorder = langsorder.strip()
  if langsorder=='' or not ('L' in what or 'D' in what):
    raise ValueError("Parameter 'langsorder' or 'what' is incorrect")
  #
  n = len(entity_list)
  # Number of entities exceeds chunksize:
  if n > chunksize:
    if debug:
      print(f"INFO: The number of entities ({n}) exceeds chunksize ({chunksize}).", sep="", file=sys.stderr)
    return doChunks(w_LabelDesc, entity_list, chunksize, what=what,
                    langsorder=langsorder, debug=debug)
  #
  values = "wd:" + " wd:".join(entity_list)
  langsorder = langsorder.replace("|", ",")
  #
  ss = ''
  qs = ''
  if 'L' in what:
    ss += ' (LANG(?label) as ?labellang) ?label'
    qs += ' ?entity rdfs:label ?label.\n'
  if 'D' in what:
    ss += ' (LANG(?description) as ?descriptionlang) ?description'
    qs += ' ?entity schema:description ?description.\n'
  #
  query = f"""SELECT ?entity {ss}
WHERE {{
  VALUES ?entity {{{values}}}
  SERVICE wikibase:label {{
    bd:serviceParam wikibase:language "{langsorder}".
    {qs}
  }}
}}
"""
  #
  if debug=='query':
    print(query, file=sys.stderr)
  #
  d = reqWDQS(query, method='POST', format="csv")
  #
  d.fillna('', inplace=True)
  d.entity = d.entity.str.slice(31)
  d.index = d.entity.values
  return d

#%% w_SearchByOccupation(Qoc, mode='entity', langsorder='', wikilangs='',
#                        chunksize=10000, debug=False):
def w_SearchByOccupation(Qoc, mode='entity', langsorder='', wikilangs='',
                         chunksize=10000, debug=False):
  """
  Return the Wikidata entities which have the occupation indicated in Qoc, the
  entity for that occupation. For example, if Qoc='Q2306091', returns the
  Wikidata entities which occupation is Sociologist, among others. Use chunked
  requests if the number of entities exceeds chunksize. Also returns the
  Wikidata class of which the entities are instances of. If parameter
  'langsorder' is '', then no labels or descriptions of the entities are
  returned, otherwise the function returns them in the language order indicated
  in 'langsorder'.

  :param Qoc: The Wikidata entity of the occupation.
  :param mode: The results you want to obtain:
   - 'entity' returns the Wikidata entities which have the occupation;
   - 'count' search in WDQS to know the number of Wikidata entities with that
      occupation;
   - 'wikipedias' also the Wikipedia page of the entities are returned.
  :param langsorder: Order of languages in which the information will be
         returned, separated with '|'. If no information is given in the first
         language, next is used. If langsorder=='', then labels or descriptions
         are not returned.
  :param wikilangs: List of languages of Wikipedias to limit the search, using
         "|" as separator. Wikipedias page titles are returned in same order as
         languages in this parameter. If wikilangs='' the function returns
         Wikipedia page titles of entities in any language, not sorted. This
         parameter only is applied if mode='wikipedias'.
  :param chunksize: If the number of entities in the database or authorities'
         catalog exceeds this number, then query are made in chunks. Please,
         decrease the default value if error is raised.
  :return A data-frame with 'entity', 'entityLabel', 'entityDescription',
          'instanceof' and 'instanceofLabel' columns. Index of the data-frame
          is also set to the list of entities found.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :example
  >>> w_SearchByOccupation(Qoc="Q2306091", mode='count')
  >>> d = w_SearchByOccupation(Qoc="Q2306091", langsorder="")
  >>> d.iloc[:3]
              entity instanceof
  Q106694    Q106694         Q5
  Q1065595  Q1065595         Q5
  Q1065810  Q1065810         Q5
  >>> d = w_SearchByOccupation(Qoc="Q2306091", langsorder="en|es|fr")
  >>> d.iloc[:3]
              entity       entityLabel  ... instanceof instanceofLabel
  Q1084790  Q1084790  Christoph Ehmann  ...         Q5           human
  Q1085054  Q1085054    Christoph Görg  ...         Q5           human
  Q1085476  Q1085476  Christoph Maeder  ...         Q5           human
  >>> d = w_SearchByOccupation(Qoc="Q2306091", mode='wikipedias',
                               wikilangs='en|es|fr', debug='info')
  """
  # First: known the number of entities
  query = 'SELECT (COUNT(DISTINCT ?entity) AS ?count) WHERE {?entity wdt:P106 wd:'+Qoc+'}'
  d = reqWDQS(query, method='GET', format="csv")
  nq = int(d['count'][0])
  if debug!=False:
    print(f"INFO: The number of entities with that occupation is {nq}.", file=sys.stderr)
  if mode=='count':
    return(nq)
  #
  langsorder = langsorder.strip()
  if langsorder == '':
    ss1 = sq = ss2 = ''
  else:
    ss1 = "?entityLabel ?entityDescription"
    ss2 = "(GROUP_CONCAT(DISTINCT ?instancLabel; separator='|') as ?instanceofLabel)"
    langsorder = langsorder.replace('|', ',')
    sq = f"""SERVICE wikibase:label {{bd:serviceParam wikibase:language "{langsorder}".
      ?entity rdfs:label ?entityLabel.
      ?entity schema:description ?entityDescription.
      ?instanc rdfs:label ?instancLabel.}}"""
  #
  nlim = nq//chunksize
  if nlim>0 and debug:
    print(f"INFO: The number of entities ({nq}) exceeds chunksize ({chunksize}).", file=sys.stderr)
  for k in range(nlim + 1):
    offset = chunksize*k
    if nlim>0 and debug:
      print(f" INFO: Requesting elements from {offset+1} to {min(offset+chunksize, nq)}", end="",file=sys.stderr)
    t0 = time()
    query = f"""SELECT DISTINCT ?entity {ss1}
(GROUP_CONCAT(DISTINCT ?instanc; separator='|') as ?instanceof)
{ss2}
WITH {{
    SELECT DISTINCT ?entity
    WHERE {{?entity wdt:P106 wd:{Qoc}.}}
    ORDER BY ?entity
    LIMIT {chunksize} OFFSET {offset}
    }} AS %results
WHERE {{
   INCLUDE %results.
   {sq}
   OPTIONAL {{?entity wdt:P31 ?instanc.}}
}} GROUP BY ?entity {ss1}
"""
    #
    if debug=='query':
      print(query, file=sys.stderr)
    #
    d = reqWDQS(query, method="GET", format="csv")
    #
    if debug:
      print(f" ({time()-t0:.2f} seconds)", file=sys.stderr)
    #
    d.fillna('', inplace=True)
    d[['entity','instanceof']] = d[['entity','instanceof']].replace(to_replace='http://www.wikidata.org/entity/', value='', regex=True)
    d.index = d.entity.values
    if k==0:
      output = d
    else:
      output = pd.concat([output, d])
     #
  if mode=='wikipedias':
    if debug!=False:
      print("INFO: Searching for Wikipedias.", file=sys.stderr)
    w = w_Wikipedias(output.entity, wikilangs=wikilangs, debug=debug)
    output = pd.concat([output, w.iloc[:,2:]], axis=1)
  #
  return output


#%% w_SearchByIdentifiers(id_list, Pproperty, langsorder='', chunksize=3000,
#                        debug=False)
def w_SearchByIdentifiers(id_list, Pauthority, langsorder='', chunksize=3000, debug=False):
  """
  Search for entities that can match identifiers in a database or authotities'
  catalog. The identifiers are in id_list. The database or authorities'
  catalog to which these identifiers belong must be provided in parameter
  `Pauthority`. If parameter langsorder='', then no labels or descriptions of
  the entities are returned, otherwise the function returns them in the
  language order indicated in `langsorder`. Duplicated entities are deleted
  before search. Index of the data-frame returned are also set to id_list.
  Uses `doChunks` function if the number of identifier is greather that the
  value of 'chunksize' parameter.

  :param id_list: One identifier or a list of identifiers.

  :param Pauthority: Wikidata property identifier of the database or
         authorities' catalog. For example, if Pauthority = "P4439", then the
         function searches for entities that have the identifiers in the MNCARS
         (Museo Nacional Centro de Arte Reina Sofía) database. Following
         library abbreviations for the databases can be also used in the
         parameter 'Pauthority':

         library  : VIAF, LC,   BNE , ISNI, JPG,  ULAN, BNF,  GND, DNB,
         Pauthority: P214, P244, P950, P213, P245, P245, P268, P227,P227,

         library  : SUDOC, NTA,  J9U,   ELEM,  NUKAT, MNCARS
         Pauthority: P269, P1006, P8189, P1565, P1207, P4439

  :param langsorder: Order of languages in which the information will be
         returned. If langsorder='', then no labels or descriptions are
         returned.
  :param chunksize: If the number of entities exceeds this number, several
         requests will be made. This is the maximum number of entities
         requested in each request. Please decrease the default value if an
         error occurs.
  :return A Pandas data-frame with columns: 'id', 'entity', 'entityLabel',
          'entityDescription', 'instanceof' and 'instanceofLabel'.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :example

  >>> d = w_SearchByIdentifiers(["4938246", "36092166", "40787112"], Pauthority='P214')
  >>> d
                  id    entity instanceof
  4938246    4938246   Q381800         Q5
  36092166  36092166   Q712609         Q5
  40787112  40787112  Q1252859         Q5

  >>> d = w_SearchByIdentifiers(["4938246", "36092166", "40787112"],
                                langsorder='en|es', Pauthority='P214')
  >>> d
                  id    entity     entityLabel   entityDescription instanceof instanceofLabel
  40787112  40787112  Q1252859   Rafael Aburto   Spanish architect         Q5           human
  36092166  36092166   Q712609  Max Abramovitz  American architect         Q5           human
  4938246    4938246   Q381800       Paul Adam     French novelist         Q5           human
  """
  #
  if isinstance(id_list, str):
    id_list = [id_list]
  id_list = [x.strip() for x in id_list if not re.match('\s*$', x)]
  if len(id_list) == 0:
    raise ValueError("Invalid value for parameter 'id_list'")
  # Remove duplicates preserving order
  id_list = list(dict.fromkeys(id_list))
  #
  libraries = {
    'VIAF':   'P214',   'LC':      'P244',  'BNE':   'P950',
    'ISNI':   'P213',   'JPG':     'P245',  'ULAN':  'P245',
    'BNF':    'P268',   'GND':     'P227',  'DNB':   'P227',
    'SUDOC':  'P269',   'idRefID': 'P269',  'NTA':   'P1006',
    'J9U':    'P8189',  'ELEM':   'P1565',  'NUKAT': 'P1207',
    'RERO':   'P3065',  'CAOONL': 'P8179',  'NII':   'P4787',
    'BIBSYS': 'P1015',  'NORAF' : 'P1015',  'BNC':   'P9984',
    'CANTIC': 'P9984',  'PLWABN': 'P7293',  'NLA' :  'P409',
    'MNCARS': 'P4439'
    }

  # Obtain de Pauthority if it is an abreviation of the library.
  m = re.match(r'^P\d+$', Pauthority)
  if m is None:
    if Pauthority not in libraries:
      raise ValueError(f"Invalid value '{Pauthority}' for parameter 'Pauthority'")
    Pauthority = libraries[Pauthority]
  #
  n = len(id_list)
  # Number of entities exceeds chunksize:
  if n > chunksize:
    if debug:
      print(f"INFO: The number of entities ({n}) exceeds chunksize ({chunksize}).", sep="", file=sys.stderr)
    return doChunks(w_SearchByIdentifiers, id_list, chunksize,
                    Pauthority=Pauthority, langsorder=langsorder, debug=debug)
  #
  if langsorder == '':
    ss1 = sq = ss2 = ''
  else:
    ss1 = "?entityLabel ?entityDescription"
    ss2 = "(GROUP_CONCAT(DISTINCT ?instancLabel; separator='|') as ?instanceofLabel)"
    langsorder = langsorder.replace('|', ',')
    sq = f"""SERVICE wikibase:label {{bd:serviceParam wikibase:language "{langsorder}".
      ?entity rdfs:label ?entityLabel.
      ?entity schema:description ?entityDescription.
      ?instanc rdfs:label ?instancLabel.}}"""
  #
  langsorder = langsorder.replace("|", ",")
  values = '"' + '" "'.join(id_list) + '"'
  #
  query = f"""
SELECT DISTINCT ?id ?entity {ss1}
(GROUP_CONCAT(DISTINCT ?instanc; separator='|') as ?instanceof)
{ss2}
WHERE {{
  OPTIONAL {{
    VALUES ?id {{{values}}}
    OPTIONAL {{?entity wdt:{Pauthority} ?id;
                      wdt:P31 ?instanc.}}
    {sq}
  }}
}} GROUP BY ?id ?entity {ss1}
"""
  if debug=='query':
    print(query, file=sys.stderr)
  #
  d = reqWDQS(query, method='POST', format="csv")
  #
  d.fillna('', inplace=True)
  d[['entity','instanceof']] = d[['entity','instanceof']].replace(to_replace='http://www.wikidata.org/entity/', value='', regex=True)
  d.index = d['id'].values
  return d

#%% w_SearchByAuthority(Pauthority, langsorder='', instanceof='', chunksize=10000,
#                        debug=False)
def w_SearchByAuthority(Pauthority, langsorder='', instanceof='', chunksize=10000,
                         debug=False):
  """
  Get all Wikidata entities that have identifiers in the database or
  authorities' catalog indicated in the parameter 'Pauthority'. Returns the
  Wikidata entities. If parameter 'langsorder'== '', then no labels or
  descriptions of the entities are returned, otherwise the function returns
  them in the language order indicated in 'langsorder'.

  :param Pauthority: Wikidata property identifier of the database or
         authorities' catalog. For example, if Pauthority = "P4439", all
         entities which have an identifier in the MNCARS (Museo Nacional Centro
         de Arte Reina Sofía) database are returnd. Following libraries
         abbreviation for the databases can be also used in the parameter
         'Pauthority':

         library  : VIAF, LC,   BNE , ISNI, JPG,  ULAN, BNF,  GND, DNB,
         Pauthority: P214, P244, P950, P213, P245, P245, P268, P227,P227,

         library  : SUDOC, NTA,  J9U,   ELEM,  NUKAT, MNCARS
         Pauthority: P269, P1006, P8189, P1565, P1207, P4439

  :param langsorder: Order of languages in which the information will be
         returned, separated with '|'. If no information is given in the first
         language, next is used. If langsorder='', then labels or descriptions
         are not returned.
  :param instanceof: Wikidata entity of which the entities searched for are an
         example or member of it (class). Optional. For example, if
         instanceof="Q5" the search are filtered to Wikidata entities of class
         Q5 (human). Some entity classes are allowed, separated with '|'.
  :param chunksize: If the number of entities in the database or authorities'
         catalog exceeds this number, then query are made in chunks. The value
         can increase if langorder=''. Please, decrease the default value if
         error is raised.
  :return A Pandas data-frame with columns: 'entity', 'entityLabel',
         'entityDescription', 'instanceof', instanceofLabel' and the
         identifier in the "Pauthority" database.
         Index of the data-frame is also set to the list of entities found.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :example
  >>> # P9944: Database of Czech Amateur Theater person ID
  ... d = w_SearchByAuthority(Pauthority="P9944", langsorder="")
  >>> d.iloc[:3]
                entity instanceof P9944
  Q949091      Q949091         Q5  3355
  Q955415      Q955415         Q5  1639
  Q68002114  Q68002114         Q5  5390
  >>> mncars = w_SearchByAuthority(Pauthority="MNCARS", langsorder="en|es")
  >>> mncars.iloc[:3]
                entity           entityLabel  ... instanceofLabel            P4439
  Q5799578    Q5799578           Darío Urzay  ...      ser humano      urzay-dario
  Q94502963  Q94502963       Jorge Ballester  ...      ser humano  ballester-jorge
  Q729079      Q729079  José Hernández Muñoz  ...      ser humano   hernandez-jose
  >>> # Not human
  ... noQ5 = mncars[~ mncars.instanceof.str.contains(r'\bQ5\b')]
  >>> # filter human
  ... mncarsQ5 = w_SearchByAuthority(Pauthority="MNCARS", langsorder = 'es|en',
                                     instanceof='Q5')
  """
  #
  libraries = {
    'VIAF':   'P214',   'LC':      'P244',  'BNE':   'P950',
    'ISNI':   'P213',   'JPG':     'P245',  'ULAN':  'P245',
    'BNF':    'P268',   'GND':     'P227',  'DNB':   'P227',
    'SUDOC':  'P269',   'idRefID': 'P269',  'NTA':   'P1006',
    'J9U':    'P8189',  'ELEM':   'P1565',  'NUKAT': 'P1207',
    'RERO':   'P3065',  'CAOONL': 'P8179',  'NII':   'P4787',
    'BIBSYS': 'P1015',  'NORAF' : 'P1015',  'BNC':   'P9984',
    'CANTIC': 'P9984',  'PLWABN': 'P7293',  'NLA' :  'P409',
    'MNCARS': 'P4439'
    }
  # Obtain de Pauthority if it is an abreviation of the library.
  m = re.match(r'^P\d+$', Pauthority)
  if m is None:
    if Pauthority not in libraries:
      raise ValueError(f"Invalid value '{Pauthority}' for parameter 'Pauthority'")
    Pauthority = libraries[Pauthority]
  #
  if langsorder == '':
    ss1 = sq = ss2 = ''
  else:
    ss1 = "?entityLabel ?entityDescription"
    ss2 = "(GROUP_CONCAT(DISTINCT ?instancLabel; separator='|') as ?instanceofLabel)"
    langsorder = langsorder.replace('|', ',')
    sq = f"""SERVICE wikibase:label {{bd:serviceParam wikibase:language "{langsorder}".
      ?entity rdfs:label ?entityLabel.
      ?entity schema:description ?entityDescription.
      ?instanc rdfs:label ?instancLabel.}}"""
  #
  if instanceof!='' and debug!=False:
    print(f"INFO: The instanceof filtering ({instanceof}) will be applied when all entities are retrieved", file=sys.stderr)
    #
  # First: known the number of entities
  query = f"SELECT (COUNT(DISTINCT ?entity) AS ?count) WHERE {{?entity wdt:{Pauthority} [].}}"
  d = reqWDQS(query, method='GET', format="csv")
  nq = int(d['count'][0])
  #
  nlim = nq//chunksize
  if nlim>0 and debug:
    print(f"INFO: The number of entities ({nq}) exceeds chunksize ({chunksize}).", file=sys.stderr)
  for k in range(nlim + 1):
    offset = chunksize*k
    if nlim>0 and debug:
      print(f"\tINFO: Requesting elements from {offset+1} to {min(offset+chunksize, nq)}", end="",file=sys.stderr)
    t0 = time()
    query = f"""SELECT DISTINCT ?entity {ss1}
(GROUP_CONCAT(DISTINCT ?instanc; separator='|') as ?instanceof)
{ss2}
(GROUP_CONCAT(DISTINCT STR(?authid);separator='|') as ?{Pauthority})
WITH {{
  SELECT DISTINCT ?entity ?authid WHERE {{?entity wdt:{Pauthority} ?authid.}}
  ORDER BY ?entity
  LIMIT {chunksize} OFFSET {offset}
  }} AS %results
WHERE {{
  INCLUDE %results.
  {sq}
  OPTIONAL {{?entity wdt:P31 ?instanc.}}
}}GROUP BY ?entity {ss1}
"""
    #
    if debug=='query':
      print(query, file=sys.stderr)
    #
    d = reqWDQS(query, method="GET", format="csv")
    #
    if nlim>0 and debug:
      print(f" ({time()-t0:.2f} seconds)", file=sys.stderr)
    #
    d.fillna('', inplace=True)
    d[['entity','instanceof']] = d[['entity','instanceof']].replace(to_replace='http://www.wikidata.org/entity/', value='', regex=True)
    d.index = d.entity.values
    if k==0:
      output = d
    else:
      output = pd.concat([output, d])
     #
  # Filter instanceof
  if instanceof!='':
    output = output[output.instanceof.str.contains(r'\b(?:' + instanceof + r")\b")]
  #
  return output

#%% def w_SearchByInstanceof(instanceof, langsorder='', chunksize=2500, debug=False):
def w_SearchByInstanceof(instanceof, langsorder='', chunksize=2500, debug=False):
  """
  Get all Wikidata entities which are instance of one o more Wikidata entities
  like films, cities, etc. If parameter `langsorder`='', then no labels or
  descriptions of the entities are returned, otherwise the function returns
  them in the language order indicated in `langsorder`.

  :param instanceof: Wikidata entity of which the entities searched for are an
         example or member of it (class). For example, if instanceof="Q229390"
         return Wikidata entities of class Q229390 (3D films). More than one
         entities can be included in the `instanceof` parameter, with '|' or
         '&' separator:
         - if '|' (instanceof='Q229390|Q202866') then the OR operator is used.
         - if '&' (instanceof='Q229390|Q202866') then the AND operator is used.

         Note that '|' and '&' cannot be present at the same time.

  :param langsorder: Order of languages in which the information will be
         returned, separated with '|'. If no information is given in the first
         language, next is used. If langsorder=='', then labels or descriptions
         are not returned.
  :param chunksize: If the number of entities which are instace of `instanceof`
         paramenter exceeds this number, then query are made in chunks. The
         value can increase if langorder=''. Please, reduce the default value
         if error is raised.
  :param debug: For debugging purposes (default FALSE). If debug='info'
         information about chunked queries is shown. If debug='query' also the
         query launched is shown. If debug='count' the function only returns
         the number of entities.
  :return A dataframe. Index of the data-frame is also set to the list of
          entities found.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :examples
  >>> w = w_SearchByInstanceof('Q229390', langsorder = 'es|en')
  >>> w = w_SearchByInstanceof('Q229390|Q202866', langsorder = 'es|en')
  >>> w = w_SearchByInstanceof('Q229390&Q202866', langsorder = 'es|en')
  """
  #
  searchOR = searchAND = False
  m = re.search("^Q\d+([|&]Q\d+)*$", instanceof)
  if m is None:
    raise ValueError(f"Invalid value '{instanceof}' for parameter 'instanceof'")
  if '|' in instanceof:
    searchOR = True
  if '&' in instanceof:
    searchAND = True
  if searchOR and searchAND:
    raise ValueError(f"ERROR: Invalid value '{instanceof}' for parameter 'instanceof'")
  #
  if langsorder == '':
    ss1 = sq = ss2 =  ''
  else:
    ss1 = "?entityLabel ?entityDescription"
    ss2 = "\n(GROUP_CONCAT(DISTINCT ?instancLabel; separator='|') as ?instanceofLabel)"
    langsorder = langsorder.replace("|", ",")
    sq = f"""\nSERVICE wikibase:label {{bd:serviceParam wikibase:language "{langsorder}".
    ?entity rdfs:label ?entityLabel.
    ?entity schema:description ?entityDescription.
    ?instanc rdfs:label ?instancLabel.}}"""
  #
  # queryC: known the number of entities
  # queryF: retrieve entites in chunks (LIMIT .. OFFSET ..)
  if searchOR:
    iofs = instanceof.split('|')
    values = 'wd:' +  " wd:".join(iofs)
    queryC = "SELECT (COUNT(DISTINCT ?entity) AS ?count) WHERE {VALUES ?iof {" + values + "} ?entity wdt:P31 ?iof}\n"
    queryF = "SELECT DISTINCT ?entity WHERE {VALUES ?iof {" + values + "} ?entity wdt:P31 ?iof}"
  elif searchAND:
    iofs = instanceof.split('&')
    values = 'wd:' + ",wd:".join(iofs)
    queryC = "SELECT (COUNT(DISTINCT *) AS ?count) WHERE {[] wdt:P31 " + values + "}\n"
    queryF = "SELECT DISTINCT ?entity WHERE {?entity wdt:P31 " + values + "}"
  else:
    queryC = "SELECT (COUNT(DISTINCT *) AS ?count) WHERE {[] wdt:P31 wd:" + instanceof + "}\n"
    queryF = "SELECT DISTINCT ?entity WHERE {?entity wdt:P31 wd:" + instanceof + "}"
  #
  if debug=="query":
    print(queryC, file=sys.stderr)
    #
  d = reqWDQS(queryC, method='GET', format="csv")
  nq = int(d['count'][0])
  #
  if debug!=False:
    print(f"INFO: The number of entities is {nq}.", file=sys.stderr)
    #
  if debug=='count':
    return(nq)
    #
  nlim = int(nq/chunksize)
  if nlim>0 and debug!=False:
    print(f"INFO: The number of entities ({nq}) exceeds chunksize ({chunksize})", file=sys.stderr)
  for k in range(nlim + 1):
    offset = k*chunksize
    if (offset+1) > nq:
      break
    if nlim>0 and debug!=False:
      t0 = time()
      print(f"  INFO: Requesting elements from {offset+1} to {min(offset+chunksize, nq)}", end="", file=sys.stderr)
    #
    query = f"""SELECT DISTINCT ?entity {ss1}
(GROUP_CONCAT(DISTINCT ?instanc; separator='|') as ?instanceof){ss2}
WITH {{
  {queryF}
  ORDER BY ?entity
  LIMIT {chunksize} OFFSET {offset}
  }} AS %results
WHERE {{
  INCLUDE %results.
  OPTIONAL {{?entity wdt:P31 ?instanc.}}{sq}
}} GROUP BY ?entity {ss1}"""
    #
    if debug=="query":
      print(query, file=sys.stderr)
      #
    d = reqWDQS(query, format='csv', method='POST')
    d.fillna('', inplace=True)
    d[['entity','instanceof']] = d[['entity','instanceof']].replace(to_replace='http://www.wikidata.org/entity/', value='', regex=True)
    d.index = d.entity.values
    #
    if nlim>0 and debug!=False:
      print(f" ({time() - t0:.2f} seconds)", file=sys.stderr )
      #
    if k==0:
      output = d
    else:
      output = pd.concat([output, d])
  #
  return output



#%% w_SearchByLabel(string, mode='inlabel', langs='', langsorder='', instanceof="",
#                    Pproperty="",debug=False):
def w_SearchByLabel(string, mode='inlabel', langs='', langsorder='', instanceof="",
                     Pproperty="", debug=False):
  """
  Search Wikidata entities by string (usually labels). Search the 'string' in
  label and altLabel ("Also known as") or in any part of the Wikidata entities
  using different approaches.

  :param string: String (label or altLabel) to search. Note that single
         quotation mark must be escaped (string="O\\'Donell"), otherwise an
         error will be raised.
  :param mode: The mode to perform search. Default 'inlabel' mode.
   - 'exact' for an exact search in label or altLabel using case sensitive
     search and differentiate diacritics. Languages in the parameter `lang` are
     used, so this parameter is mandatory using this mode.
   - 'startswith' for entities which label or altLabel starts with the string,
     similar to a wildcard search "string*". The string is searched in label in
     the languages of `lang` parameter, but in any language in altLabel, so
     parameter `lang` is also mandatory in this mode. Diacritics and case are
     ignored in this mode.
   - 'cirrus' search words in any order in any part of the entity (which must
     be a string), not only in label or altLabel. Diacritics and case are
     ignored. It is a full text search using the ElasticSearch engine.
     Phrase search can be used if launched with double quotation marks, for
     example, string='"Antonio Saura"'. Also fuzzy search is possible, for
     example, string="algermon~1" or string="algernon~2". Also REGEX search can
     be used (but it is a very limited functionality) using this format:
     string="insource:/regex/i" (i: is for ignore case, optional).
     In this mode, parameter `langs` is ignored.
   - 'inlabel' is an special case of 'cirrus' search for matching whole words
     (in any order) in any position in label or altLabel. With this mode no
     fuzzy search can be used, but some languages can be set in the `lang`
     parameter.
     Modes 'inlabel' and 'cirrus' use the CirrusSearch of the Wikidata API.
     Please, for more examples, see https://www.mediawiki.org/wiki/Help:CirrusSearch
     and https://www.mediawiki.org/wiki/Help:Extension:WikibaseCirrusSearch
  :param langs: Languages in which the information will be searched, using "|"
         as separator. In 'exact' or 'startswith' modes this parameter is
         mandatory, at least one language is required. In 'inlabel'mode, if the
         parameter `langs`   is set, then the search is restricted to languages
         in this parameter, otherwise any language. In 'cirrus' mode this p
         arameter is ignored.
  :param langsorder: Order of languages in which the information will be
         returned, using "|" as separator. If `langsorder`='', no labels or
         descriptions will be returned, otherwise, they are returned in the
         order of languages in this parameter, if any.
  :param instanceof: Wikidata entity of which the entities searched for are an
         example or member of it (class). For example, if instanceof='Q5' the
         search are filtered to Wikidata entities of class Q5 (human). Some
         entity classes are allowed, separated with '|'.
  :param debug For debugging purposes (default FALSE). If debug='query' the
         query launched is shown. If debug='count' the function only returns
         the number of entities with that occupation.
  :return: A Pandas data-frame with columns: 'entity', 'entityLabel',
         'entityDescription', 'instanceof' and 'instanceofLabel'.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :example
  >>> df = w_SearchByLabel(string='Iranzo', mode="exact", langs='es|en')
  >>> df = w_SearchByLabel(string='Iranzo', mode="exact", langs='es|en', langsorder='es|en', instanceof = 'Q5|Q101352')
  >>> ## Search entities which label or altLabel starts with "string"
  ... df = w_SearchByLabel(string='Iranzo', mode='startswith', langs='en', langsorder='es|en')
  >>> ## Search in any position in Label or AltLabel (diacritics and case are ignored)
  ... df = w_SearchByLabel(string='Iranzo', mode='inlabel', langsorder='es|en')
  >>> ## Search in Chinese (Simplified) (language code: zh) in any part of entity:
  ... df = w_SearchByLabel(string='伊兰佐', mode='cirrus', langsorder='es|zh|en')
  >>> d = w_SearchByLabel('Antonio Saura', mode='exact', langs='en|es')
  >>> d.iloc[:3]
                  entity          instanceof
  Q528511        Q528511                  Q5
  Q3620165      Q3620165  Q2175765|Q22808404
  Q104866130  Q104866130                  Q5
  >>> d = w_SearchByLabel('Antonio Saura', mode='inlabel', langs='', langsorder='en|es')
  >>> len(d)
  14
  >>> d = w_SearchByLabel('"Antonio Saura"', mode='inlabel', langs='', langsorder='en|es')
  >>> len(d)
  12
  >>> d = w_SearchByLabel('Antonio Saura', mode='exact', langs='en|es', Pproperty='P21')
  """
  #
  # string = string.strip().replace("'", r"\'")
  langs = langs.strip()
  string = string.strip()
  #
  if string=='' or mode not in ['exact', 'startswith', 'inlabel', 'cirrus']:
    raise ValueError("Parameter 'string' or 'mode' is invalid.")
  if mode in ['exact', 'startswith'] and langs=='':
    raise ValueError(f"Error: parameter 'langs' is mandatory in mode '{mode}'")
  if mode=='cirrus' and langs!='':
    print(f"INFO: in mode {mode} parameter 'langs' is ignored", file=sys.stderr)
  #
  p_show = p_query = p_lang = ''
  if Pproperty!='':
    p_show  = []
    p_query = []
    p_lang  = []
    for p in Pproperty.split('|'):
      p_show.append(f"(GROUP_CONCAT(DISTINCT ?{p}p;separator='|') as ?{p})")
      if langsorder!='':
        p_show.append(f"(GROUP_CONCAT(DISTINCT STR(?{p}label);separator='|') as ?{p}Label)")
      p_query.append(f"OPTIONAL {{?entity wdt:{p} ?{p}p.}}")
      p_lang.append(f"?{p}p rdfs:label ?{p}label.")
    p_show  = '\n'.join(p_show)
    p_query = '\n'.join(p_query)
    p_lang  = '\n'.join(p_lang)

  if langsorder == '':
    ss1 = ''
    ss2 = p_show
    sq  = p_query
  else:
    ss1 = "?entityLabel ?entityDescription"
    ss2 = "(GROUP_CONCAT(DISTINCT ?instancLabel; separator='|') as ?instanceofLabel)"
    ss2 += "\n"+p_show
    langsorder = langsorder.replace('|', ',')
    sq = f"""{p_query}\n  SERVICE wikibase:label {{bd:serviceParam wikibase:language "{langsorder}".
      ?entity rdfs:label ?entityLabel.
      ?entity schema:description ?entityDescription.
      ?instanc rdfs:label ?instancLabel.
      {p_lang}}}"""
  #
  # Start of the query
  query = f"""SELECT DISTINCT ?entity {ss1}
(GROUP_CONCAT(DISTINCT ?instanc; separator='|') as ?instanceof)
{ss2}
"""
  #
  if mode=='exact':   # UNION sentences with each language
    unionlabel    = '\nUNION\n'.join([f' {{?entity rdfs:label "{string}"@{l}}}' for l in langs.split('|')])
    unionaltLabel = '\nUNION\n'.join([f' {{?entity skos:altLabel "{string}"@{l}}}' for l in langs.split('|')])
    #
    query += f"""WHERE {{
  {unionlabel}
  UNION
  {unionaltLabel}
"""
  #
  elif mode=='startswith':
    mwapi = []
    langs = langs.split('|')
    for l in langs:
      mwapi.append(f"""{{
       SERVICE wikibase:mwapi {{
         bd:serviceParam wikibase:api "EntitySearch";
                         wikibase:endpoint "www.wikidata.org";
                         mwapi:language "{l}";
                         mwapi:search "{string}".
         ?entity wikibase:apiOutputItem mwapi:item.}}
      }}""")
    mwapi = '\n    UNION\n    '.join(mwapi)
    #
    query += f"""WITH {{
    SELECT DISTINCT ?entity
    WHERE {{
      {mwapi}
    }}
  }} AS %results
  WHERE {{
    INCLUDE %results"""
  #
  elif mode in ['inlabel', 'cirrus']:
    if mode=='inlabel':
      string = 'inlabel:' + string
      if langs != '':
        string += '@' + langs.replace('|',',')
    query += f"""WHERE {{
  SERVICE wikibase:mwapi {{
    bd:serviceParam wikibase:api "Search";
                    wikibase:endpoint "www.wikidata.org";
                    mwapi:srsearch '{string}'.
    ?entity wikibase:apiOutputItem mwapi:title.
  }}"""
  #
  else:
    raise ValueError("Invalid value for parameter 'mode'")
  #
  # The final part of the query:
  query += f"""
  OPTIONAL {{?entity wdt:P31 ?instanc.}}
  {sq}
}} GROUP BY ?entity {ss1}"""
  #
  if debug:
    print(query, file=sys.stderr)
  #
  d = reqWDQS(query, method="GET", format="csv")
  #
  d.fillna('', inplace=True)
  d.entity = d.entity.str.slice(31)
  d.instanceof = d.instanceof.replace(to_replace='http://www.wikidata.org/entity/', value='', regex=True)
  #
  if Pproperty!='':
    for p in Pproperty.split('|'):
      d[p] = d[p].replace(to_replace='http://www.wikidata.org/entity/', value='', regex=True)
  #
  d.index = d.entity.values
  # Filter instanceof
  if instanceof!='':
    d = d[d.instanceof.str.contains(r'\b(?:' + instanceof + r")\b")]
  #
  return d


#%% w_EntityInfo(entity_list, langsorder='en', wikilangs="", debug=False):
def w_EntityInfo(entity_list, mode='human', langsorder='', wikilangs="",
                  chunksize=MW_LIMIT, debug=False):
  """
  Get information about a Wikimedia entity (human or film).
  This function uses the WikiBase API to obtain information from Wikidata (this
  API is similar to the MediaWiki API, so uses the reqMediaWiki() function) and
  the Wikidata Query Service (also uses the reqWDQS() function).

  Get labels, descriptions and some properties of the Wikidata entities in
  entity_list, for person or films. If person, the information returned is
  about labels, descriptions, birth and death dates and places, occupations,
  works, education sites, awards, identifiers in some databases, Wikipedia page
  titles (which can be limited to the languages in the `wikilangs` parameter,
  etc. If films, information is about title, directors, screenwriter,
  castmember, producers, etc.

  :param entity_list: The Wikidata entities to search for properties (person or
         films.)
  :param mode: In "human" mode, the list of entities is expected to
         correspond to people, obtaining information about people. If the
         mode is "film", information related to films will be requested.
  :param langsorder: Order of languages in which the information will be
         returned, separated with '|'. If no information is given in the first
         language, next is used. For label and description, English is used for
         language failback, if they are not in English, then information is
         returned in any else language. The language for label and description
         are also returned. If langsorder=='', then no other information than
         labels or descriptions are returned in any language, only Wikidata
         entities, else, use the order in this parameter to retrieve
         information.
  :param wikilangs: List of languages to limit the search of Wikipedia pages,
         using "|" as separator. Wikipedias pages are returned in same order as
         languages in this parameter. If wikilangs='' the function returns
         Wikipedia pages in any language, not sorted.
  :param debug: For debugging (info or query)
  :return: A data-frame with the properties of the entities, also the index
         of the dataframe is set to the list of entities.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :examples
  >>> df = w_EntityInfo(entity_list='Q134644', langsorder='es|en')
  >>> df = w_EntityInfo(entity_list='Q134644', langsorder='es|en',
                        wikilangs='es|en|fr')
  >>> df = w_EntityInfo(['Q270510', 'Q1675466', 'Q24871'], mode='film',
                        langsorder='es|en', wikilangs='es|en|fr')
  >>> # Search string 'abba' inlabel, instanceof='Q5'
  ... w = w_SearchByLabel('abba', mode='inlabel', langsorder = '', instanceof = 'Q5')
  ... len(w)
  310
  >>> df = w_EntityInfo(w.entity, langsorder='en', wikilangs='en|es|fr', debug='info')
  >>> # Search 3D films: instanceof='Q229390'
  ... w = w_SearchByInstanceof(instanceof='Q229390', langsorder = 'en|es',
                               debug = 'info')
  ... df = w_EntityInfo(w.entity, mode="film", langsorder='en', wikilangs='en',
                        debug='info')

  """
  # Check entity_list
  entity_list = checkEntities(entity_list)

  if mode=='film':
    # For these claims, more than one ocurrences separated with '|' are token, except
    # for fields in fieldsonlyone, in which only the most referenced is taken
    fields = {
      'P31'  : 'instanceof',         'P577' : 'pubdate',      'P3383': 'poster',
      'P18'  : 'pic',                'P10'  : 'video',        'P1476': 'title',
      'P2047': 'duration',           'P144' : 'basedon',      'P135' : 'movement',
      'P136' : 'genre',              'P495' : 'country',      'P364' : 'originallanguage',
      'P57'  : 'director',           'P58'  : 'screenwriter', 'P161' : 'castmember',
      'P725' : 'voiceactor',         'P1431': 'executiveproducer',
      'P344' : 'photographdirector', 'P1040': 'filmeditor',
      'P2554': 'productiondesigner', 'P86'  : 'composer',     'P162' : 'producer',
      'P272' : 'productioncompany',  'P462' : 'color',        'P180' : 'depicts',
      'P921' : 'mainsubject',        'P166' : 'award',        'P444' : 'reviewscore',
      'P214' : 'VIAF',               'P480' : 'FilmAffinity', 'P345' : 'IMDb'
      }
    # For this claims, only the most referred is taken (note, however, that the
    # preferred has priority).
    fieldsonlyone = {'P577', 'P1476', 'P2047'}

    # Note that all fields that are of valuetype = "wikibase-entityid" also
    # retrieve the labels associated
    # Columns of the data-frame returned: Qentities and labels if any
    columns = """entity
      status labellang label descriptionlang description instanceofQ instanceof
      pubdate pubyear poster pic video duration title
      basedonQ basedon movementQ  movement genreQ genre countryQ country
      originallanguageQ originallanguage directorQ director
      screenwriterQ screenwriter castmemberQ castmember voiceactorQ voiceactor
      executiveproducerQ executiveproducer photographdirectorQ photographdirector
      filmeditorQ filmeditor productiondesignerQ productiondesigner
      composerQ composer producerQ producer productioncompanyQ productioncompany
      colorQ color depictsQ depicts mainsubjectQ mainsubject awardQ award
      reviewscore VIAF FilmAffinity IMDb wikipedias""".split()
      # Note that wikipedias is added at last as new column of the dataframe
  else: # people
    # For these claims, more than one ocurrences separated with '|' are token, except
    # for fields in fieldsonlyone, in which only the most referenced is taken
    fields = {
      'P31' :'instanceof',  'P18' :'pic',        'P21'  :'sex',
      'P69' :'educatedat',  'P106':'occupation', 'P101' :'fieldofwork',
      'P135':'movement',    'P136':'genre',      'P737' :'influencedby',
      'P800':'notablework', 'P463':'memberof',   'P166' :'award',
      'P214':'viafid',      'P950':'bneid',      'P4439':'mncarsid',
      'P19' :'bplace',      'P20' :'dplace',
      'P569':'bdate',       'P570':'ddate',
      }
    # For this claims, only the most referred is taken (note, however, that the
    # preferred has priority).
    fieldsonlyone = {'P19', 'P20', 'P569', 'P570'}

    # Note that all fields that are of valuetype = "wikibase-entityid" also
    # retrieve the labels associated
    # Columns of the data-frame returned: Qentities and labels if any
    columns = """entity
      status labellang label descriptionlang description instanceofQ instanceof sexQ sex
      bdate byear bplaceQ bplace bplaceLat bplaceLon bcountryQ bcountry
      ddate dyear dplaceQ dplace dplaceLat dplaceLon dcountryQ dcountry
      occupationQ occupation notableworkQ notablework educatedatQ educatedat
      fieldofworkQ fieldofwork movementQ movement genreQ genre influencedbyQ influencedby
      memberofQ memberof awardQ award viafid bneid mncarsid pic wikipedias""".split()
      # Note that wikipedias is added at last as new column of the dataframe
  #
  # langsorder: failback: en
  langsorder = langsorder.strip()
  llangs = langsorder.split('|')  # label langs
  if 'en' not in llangs:
    llangs.append('en')
  langsorder = '|'.join(llangs)
  # wikilangs
  wlangs = []
  if wikilangs!="":
    wlangs = wikilangs.split('|')   # wiki langs
  #
  d = dict()
  qidsoflabels = set()  # store entities for searching labels
  qidsofplaces = set()  # store entities for searching places
  # Check limits to make chucked queries
  n = len(entity_list)
  for k in range(int(n/chunksize)+1):
    if n>chunksize and k==0:
      print(f"INFO: The number of entities ({n}) exceeds the MediaWiki API limit ({chunksize}): doing chunked requests.", file=sys.stderr)
    offset = chunksize*k
    t_list = entity_list[offset:offset+chunksize]
    if len(t_list) == 0:
      break
    if n>chunksize:
      print(f" INFO: Getting entities from {offset+1} to {offset+len(t_list)}", file = sys.stderr)
    #
    query = {"format" : 'json',
             "action" : 'wbgetentities',
             "props"  : 'labels|descriptions|claims|sitelinks',
             "ids"    : '|'.join(t_list)}
    #
    # Only return sitelinks if wikilangs!='' because the wikipedia sites can
    # easily be selected in the query. If wikilangs='', not only wikipedia,
    # also other projects are returned.
    if wikilangs!="":
      query["sitefilter"] = '|'.join([x+'wiki' for x in wlangs])
    #
    j = reqMediaWiki(query=query, project="www.wikidata.org", debug=debug)
    #
    if j is None or "success" not in j or j['success'] != 1:
      raise NameError("ERROR in w_EntityInfo(): reqMediaWiki returns an improper JSON.")
    #
    for qid,data in j['entities'].items():  # each qid is a dict()
      d[qid] = {x:None for x in columns}
      d[qid]['entity'] = qid
      d[qid]['status'] = 'ok'
      # Check if entity is a redirection to other entity
      if 'redirects' in data:
        qid_r = data['id']
        d[qid]['status'] = qid_r
        print(f"INFO: {qid} redirects to {qid_r}, so all information returned is about {qid_r}.", file=sys.stderr)
        if qid_r in entity_list:
          print(f" Also note that {qid_r} is in the original entity list too.", file=sys.stderr)
      # Check if entity is missing
      if 'missing' in data:
        print(f"INFO: {qid} is missing.", file=sys.stderr)
        d[qid]['status'] = 'missing'
        continue
      #
      # Get the label and the description of the entity. Retrieves the first
      # in language order or any one else, if exist, otherwise None.
      for ld in ['label', 'description']:
        item = ld + 's'    # label(s) and description(s) items
        if item not in data or len(data[item])==0:
          d[qid][ld]        = None
          d[qid][ld+"lang"] = None
          continue
        lds = data[item]
        existlang = False
        for lang in llangs:
          if lang in lds and 'for-language' not in lds[lang]:
            existlang = True
            d[qid][ld]        = lds[lang]['value']
            d[qid][ld+'lang'] = lds[lang]['language']
            break
        if not existlang:
          lang = next(iter(lds))
          d[qid][ld]        = lds[lang]['value']
          d[qid][ld+'lang'] = lds[lang]['language']
      #
      # Processing claims
      claims = data['claims']
      #
      # For f in fields list retrieve all values, but if f is in fieldsonlyone,
      # only the most referenced value is taken. For both, if there is a
      # preferred value, then, this one is taken.
      for f,fname in fields.items():
        if f in claims:
          values = []
          nrefs  = []
          is_entity = False  # If the claim store a Qxxx, add qids in the corresponding "Q" field
          for item in claims[f]:
            if 'datavalue' not in item['mainsnak']: # Unknown value in the claim, but some information exists about the property.
              continue
            valuetype = item['mainsnak']['datavalue']['type']
            value     = item['mainsnak']['datavalue']['value']
            # count references
            if 'references' in item:
              nrefs.append(len(item['references']))
            else:
              nrefs.append(0)
            # datatypes
            if valuetype == 'string':
              v = value
            elif valuetype == 'wikibase-entityid':
              is_entity = True
              v = value['id']
              # Store qids to search labels later (not for places, because labels
              # are returned when retrieve places)
              if f not in ['P19', 'P20']:
                qidsoflabels.add(v)
            elif valuetype == 'time':
              v = value['time']
            elif valuetype == 'monolingualtext':
              v = value['text'] + ':' + value['language']
            elif valuetype == 'quantity':
              unit = value['unit']
              m = re.match('http://www.wikidata.org/entity/(Q.*)', unit)
              if m is not None:
                unit = m.group(1)
                qidsoflabels.add(unit)
              v = value['amount'] + ' : ' + unit
            else:
              print(f" WARNING in mm_EntityInfo: valuetype {valuetype} not implemented .", file=sys.stderr)
            #
            # Check reviewers for P444 (review score)
            if f=='P444' and 'qualifiers' in item:
              for qualifier,dq in item['qualifiers'].items():
                if qualifier == 'P447':
                  reviewerQ = dq[0]['datavalue']['value']['id']
                  qidsoflabels.add(reviewerQ)
                  v += f' [{reviewerQ}]'
            #
            if 'rank' in item and item['rank']=='preferred':
              values = [v]
              nrefs  = [0]
              break
            else:
              values.append(v)
          #
          if len(values)!=0: # The claim almost has one not erroneous value
            if f not in fieldsonlyone:
              d[qid][fname] = '|'.join(set(values))
            else:
              # Get the most referred value: # index_max = max(range(len(nrefs)), key=nrefs.__getitem__)
              index_max = nrefs.index(max(nrefs))
              v = values[index_max]
              d[qid][fname] = v
            # Store qids to search places later
            if f in ['P19', 'P20']:
              qidsofplaces.add(v)
            # Extract year from bdate/ddate/pubdate:
            if f in ['P569', 'P570', 'P577'] and f in fields:
              ff = fname.replace("date", "year")  # bdate/ddate => byear/dyear
              d[qid][ff] = v[1:5]                 # +yyyy-MM-ddThh:mm:ssZ => yyyy
            # if is_entity add qids to the corresponding "Q" field
            if is_entity:
              d[qid][fname+"Q"] = d[qid][fname]

      # Wikipedias
      sitelinks = data['sitelinks']
      if len(sitelinks) > 0:
        wvalues = []
        if len(wlangs)==0:
          for wlwiki in sitelinks.keys():
            if wlwiki.endswith("wiki"):
              wl = wlwiki[0:-4]
              title = sitelinks[wlwiki]['title'].replace(" ","_")
              title = requests.utils.quote(title)
              wvalues.append(f'https://{wl}.wikipedia.org/wiki/{title}')
        else:
          for wl in wlangs:
            wlwiki = wl+'wiki'
            if wlwiki in sitelinks.keys():
              title = sitelinks[wlwiki]['title'].replace(" ","_")
              title = requests.utils.quote(title)
              wvalues.append(f'https://{wl}.wikipedia.org/wiki/{title}')
        if len(wvalues) > 0:
          d[qid]['wikipedias'] = '|'.join(wvalues)

      # Add URL parts to the pic/poster/video to download, if any:
      for fname in ['poster', 'pic', 'video']:
        if fname in d[qid] and d[qid][fname] is not None:
          p = [requests.utils.quote(x.replace(" ","_")) for x in d[qid][fname].split('|')]
          u = 'https://commons.wikimedia.org/wiki/Special:FilePath/'
          d[qid][fname] = '|'.join([u+x for x in p])

  # Search places for labels, coords (Latitude and Longitude) and countries
  if len(qidsofplaces) > 0:
    if debug!=False:
      print("INFO: Searching labels, latitude and longitude coordinates, and countries for places.", file=sys.stderr)
    places = w_Geoloc(qidsofplaces, langsorder=langsorder, debug=debug)
    # Set the right colnames
    places.columns =  ['placeQ', 'place', 'placeLat', 'placeLon', 'countryQ', 'country']
    # Convert as dict() to add values to the d dataframe
    places = places.to_dict(orient='index')
    # Add labels, geo-coordinates and country to entities
    for qid,data in d.items():
      for f in 'bd': # ['bplaceQ', 'dplaceQ']:
        if data[f+'placeQ'] is not None and data[f+'placeQ'] in places:
          placeQ = data[f+'placeQ']
          data.update({f+x:y for x,y in places[placeQ].items()})

  # Search labels for the rest of Qxxx entities
  if len(qidsoflabels) > 0:
    if debug!=False:
      print("INFO: Searching labels for Wikidata entities.", file=sys.stderr)
    labels = w_LabelDesc(qidsoflabels, what='L', langsorder=langsorder, debug=debug)
    labels = labels.label.to_dict()
    # Add labels to entities
    for qid,data in d.items():
      for f in data.keys():
        if data[f] is None:
          continue
        if f.endswith('Q') and f not in ['bplaceQ', 'dplaceQ', 'bcountryQ', 'dcountryQ']:
          # Select the field without the 'Q' in last position:
          ff = f[:-1]
          # Now replace the Qentities in those fields with the associated labels
          data[ff] = '|'.join([labels[x] for x in data[ff].split('|')])
        #
        if f == 'duration':  # data['duration'] = '+125 (Q7727)'
          m = re.search(': (Q\d+)$', data['duration'])
          if m is not None:
            unit = labels[m.group(1)]
            data['duration'] = re.sub(': (Q\d+)$', unit, data['duration'])
        #
        if f == 'reviewscore': # data['reviewscore']
          reviewscore = []
          for rrss in data['reviewscore'].split('|'):
            m = re.search('(Q\d+)', rrss)
            if m is not None:
              reviewscore.append(re.sub('(Q\d+)', labels[m.group(1)], rrss))
          data['reviewscore'] = '|'.join(reviewscore)

  # Finally, convert the dict of dicts to a dataframe
  df = pd.DataFrame.from_dict(d, orient='index')
  # df.fillna("", inplace=True)
  return df


#%% -- MediaWiki API ------------------------------------------------------------
# MediaWiki API: provides direct, high-level access to the data contained in
# MediaWiki databases over the web.
# See https://www.mediawiki.org/wiki/API:Main_page
# See https://en.wikipedia.org/w/api.php

#%% reqMediaWiki(query, project='en.wikipedia.org', method='GET', attempts=2,
#                debug=False)
def reqMediaWiki(query, project='en.wikipedia.org', method='GET', attempts=2, debug=False):
  """
  Use requests package to retrieve responses in JSON format using the
  MediaWiki REST API with the query search indicated in query. For MediaWiki
  requests only user_agent is necessary in the request headers (see
  https://www.mediawiki.org/wiki/API:Etiquette ) The standard and default
  output format in MediaWiki is JSON. All other formats are discouraged. The
  output format should always be specified using the request param "format"
  in the "query" request.

  :param query: A dict with de (key, values) pairs with the search.
  :param project: The Wikimedia project to search. Default en.wikipedia.org.
  :param method: The method used in the request. Default 'GET'.
      Note in https://www.mediawiki.org/wiki/API:Etiquette#Request_limit:
      "Whenever you're reading data from the web service API, you should try to
      use GET requests if possible, not POST, as the latter are not cacheable."
  :param attempts: On "ratelimited" errors, the number of times the request is
         retried using a 60 seconds interval between retries. Default 2, if
         attempts==0 no retries are done.
  :return j: The response in JSON format or None on errors. Errors can be
          produced by means of response.raise_for_status() error or because
          the number of ratelimited errors attemps is achived.

  Note: MediaWiki API has a limit of 50 titles en each query (using "|" as
        separator). If a query contains more than 50 titles this function
        show an error and returns None. Other functions in this module use the
        function m_doChunked() to make sucesive requests with a maximum of 50
        titles each.
  """
  if not isinstance(query, dict) or project.strip() == "":
    raise ValueError("Parameter 'query' or parameter 'project' or both are invalid")
  #
  if 'titles' in query:
    articles = query['titles'].split('|')
    if len(articles) > MW_LIMIT:
      raise ValueError(f"The number of articles ({len(articles)}) exceeds the MediaWiki API limit ({MW_LIMIT})")
  #
  url = "https://" + project + "/w/api.php"
  nt = 0
  while True:
    nt += 1
    if method=='GET':
      response = requests.get(url=url, params=query, headers={'user-agent': user_agent})
    elif method=='POST':
      response = requests.post(url=url, data=query, headers={'user-agent': user_agent})
    else:
      raise ValueError(f"Method '{method}' is not supported")
    #
    response.raise_for_status()
    if debug=="query":
      print("Quoted:\n", response.url, file=sys.stderr)
      print("Unquoted:\n", requests.utils.unquote(response.url), file=sys.stderr)

    j = response.json()
    #
    if debug!=False and 'warnings' in j:
      for x,y in j['warnings'].items():
        print(f"WARNINGS: {x}: {y['warnings']}")
    if 'error' not in j:
      return j
    # See https://www.mediawiki.org/wiki/API:Etiquette#Request_limit
    if j['error']['code'] == 'ratelimited':
      if nt <= attempts:
        print(f"WARNING: Ratelimited error. Sleeping {60*nt} seconds", )
        sleep(60*nt)
      else:
        raise NameError(f'{nt} ratelimited attemps achieved, aborting.')
    else:
       raise NameError(f"ERROR: reqMediaWiki: {j['error']['code']}: {j['error']['info']}")


#%% normalizedTitle(title, q)
def normalizedTitle(title, q):
  """
  Return the normalized and the redirect title (also normalized), if any, from
  the query part of the JSON response of a MediaWiki search. The response of
  the MediaWiki API query (https://www.mediawiki.org/wiki/API:Query) includes
  original page titles and possibily normalized and redirected titles, if
  the API needs to obtain them. For a original title, this function returns
  them, if any.

  :param title: The title likely to be found in q.
  :param q: The query part of the JSON response (j['query']) from a Mediawiki
         search. Note that this part contains some titles, so it is necessary
         to search the original "title" in that part.
  :return A list with the normalized and redirected page title (target, also
          normalized) found for the title. None on errors.
  """
  anorm = title
  # Is normalized (and possibly encoded)? # See https://phabricator.wikimedia.org/T29849#2594624
  # It is supposed that order is: first, NFC normalization, then, uppercase normalization.
  if 'normalized' in q:
    for nn in q['normalized']:
      # NFC normalization
      if 'fromencoded' in nn and nn['fromencoded'] and requests.utils.quote(anorm) == nn['from']:
        anorm = nn['to']
      # Uppercase normalization
      if nn['from'] == anorm:
        anorm = nn['to']
        break

  normalized = None if anorm == title else anorm
  # ¿Is it a redirect? The normalized titles is used.
  target = None
  if 'redirects' in q:
    for nn in q['redirects']:
      if nn['from'] == anorm:
        target = nn['to']
        break
  #
  return [normalized, target]

#%% checkTitles(titles)
def checkTitles(titles):
  """
  Check if titles are valid, i.e, they not contain forbidden characteres (#<>[]{}|)
  See https://en.wikipedia.org/wiki/Wikipedia:Naming_conventions_(technical_restrictions)#Forbidden_characters
  (note that "_" is excluded in this function).

  :param titles A list of titles for cheking.
  :return The titles with duplicates removed, otherwise, raise error.
  """
  if isinstance(titles, str):
    if titles.strip() == '':
      raise ValueError("Invalid value for parameter 'titles'")
    titles = [titles]
  titles = [x.strip() for x in titles if not re.match('\s*$', x)]
  if len(titles) == 0:
    raise ValueError("Invalid value for parameter 'titles'")
  for title in titles:
    for c in title:
      if c in "#<>[]|{}": # "_" excluded
        raise ValueError(f"Invalid value for parameter 'titles': article title '{title}' has forbidden character ({c}).")
  # Remove duplicates preserving order
  titles = list(dict.fromkeys(titles))
  return titles

#%% m_Search(string, mode='title', project='en.wikipedia.org',
#            profile="engine_autoselect", limit=30, debug=False)
def m_Search(string, mode='title', project='en.wikipedia.org',
             profile="engine_autoselect", limit=30, debug=False):
  """
  Search pages in the Wikimedia 'project' that contain 'string' in the
  title or in the body of the article, depending on whether mode='title' or
  mode='text', which uses the PrefixSearch API or the CirrusSearch API,
  respectively. Returns a Pandas dataframe with pages found, ordered by
  relevance and the Wikidata entity of pages.

  The function uses a "generator" of the search API, so it is possible to
  obtain additional information, in our case the "wikibase_item".

  :param string: String to search. Note that APIs performs a NFC Unicode
         normalization on search string.
  :param mode: If mode='title', then the query uses the PrefixSearch API. If
         mode='text', it uses the CirrusSearch API.

         The PrefixSearch API searchs wikimedia pages by title. Depending on
         the search engine backend, this might include typo correction,
         redirect avoidance, stop-word removal, or other heuristics.
         See https://www.mediawiki.org/wiki/API:Prefixsearch

         The CirrusSearch API does a "full text search", which is an "indexed
         search": all pages are stored in the wiki database, and all the words
         in the non-redirect pages are stored in the search database, which is
         an index to practically the full text of the wiki. The "indexed
         search" use text processing: lower-case transformation, diacritic
         reduction, eliminate stop words and stemming. In addition, this API
         allows advanced searches, because it uses ElasticSearch as a search
         engine. For example, filters can be used: 'intitle' or 'incategory',
         including regular expresions. For example: 'intitle:"Max Planck"'
         o 'intitle:/regex/i'. More options exists, please, see
         https://www.mediawiki.org/wiki/Help:CirrusSearch .
         Note that the function, when in mode='text' only one profile is used:
         the 'engine_autoselect' profile, which is used only for sortings
         results.

  :param project: The Wikimedia project, defaults "en.wikipedia.org".
  :param profile: This parameter sets the search type. If mode='title', options
         are: strict, normal, normal-subphrases, fuzzy, fast-fuzzy,
         fuzzy-subphrases, classic and engine_autoselect (default). If
         mode='text', the function not use the parameter 'profile', because
         only use "engine_autoselect" option of the "search" action of the API
         (see https://www.mediawiki.org/w/api.php?action=help&modules=query%2Bsearch )
  :param limit: Maximun number of page titles returned.
  :return: A data-frame with order (by relevance), page titles, status and
           Wikidata entities found. None if no entities are found.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  """
  #
  if string.strip() =='':
    raise NameError("The 'string' parameter must be set.")
  #
  if mode=='title':
    query = {
      "format"       : "json",
      "formatversion": "2",
      "redirects"    : '1',      #  Operative with Prefixsearch using generators
      "action"       : "query",
      "generator"    : 'prefixsearch',
      "gpsnamespace" : "0",
      "gpsprofile"   : profile,
      "gpslimit"     : ("max" if limit>500 else limit),  # in API "max" == 500
      "prop"         : "pageprops",
      "ppprop"       : 'wikibase_item|disambiguation',
      "gpssearch"    : string  }
  elif mode=='text':
    query = {
      "format"       : "json",
      "formatversion": "2",
      "action"       : "query",
      "generator"    : 'search',
      "gsrprop"      : "",
      "gsrnamespace" : "0",
      "gsrqiprofile": "engine_autoselect",
      "gsrlimit"     : ("max" if limit>500 else limit),   # in API "max" == 500
      # "srwhat"     : 'text',  # title is disable; nearmatch not returns results
      # "srenablerewrites" : "1",
      # "srinfo"     : "totalhits|suggestion|rewrittenquery",
      # "gsrsort"     : "relevance",
      'prop'         : 'pageprops',
      "ppprop"       : 'wikibase_item|disambiguation',
      "gsrsearch"    : string  }
  else:
    raise ValueError("Invalid value for parameter 'mode'")
  #
  output=[]
  #
  getlimit = False
  while(True): # While there are "continue" responses
    j = reqMediaWiki(query=query, project=project, debug=debug)
    #
    if 'query' not in j:  # There are not results.
      break
    #
    # Note that the response includes an "index" which is the order of the
    # returned in the "gsrsort" parameter (relevante by default). It is
    # important to retrieve results in that order.
    for page in j['query']['pages']:          # With formatversion=2 q["pages"] is a list
      title  = page['title']
      index  = page['index']
      entity = None
      if 'pageprops' not in page:    # No pageprops, so no wikidata_item
        status = "no_pageprops"
      elif 'wikibase_item' not in page['pageprops']:   # No wikibase_item
        status = "no_wikibase_item"
      else:
        status = "disambiguation" if 'disambiguation' in page['pageprops'] else 'OK'
        entity = page['pageprops']['wikibase_item']
      #
      output.append({'index' : index,
                     'title' : title,
                     'status': status,
                     'entity': entity})
      if len(output) >= limit:
        getlimit = True
        break
    #
    if getlimit:
      break
    # 'Continue' response?
    if 'continue' in j:
      if debug:
        print("INFO: continue response.", file=sys.stderr)
      query.update(j['continue'])
    else:
      break
  # Finally return de dict output as a Pandas data-frame, order by relevance
  if len(output) == 0:
    return None
  return pd.DataFrame.from_dict(sorted(output, key=lambda x: x['index']))



#%% m_WikidataEntity(titles, project='en.wikipedia.org', chunksize=MW_LIMIT, debug=False)
def m_WikidataEntity(titles, project='en.wikipedia.org', chunksize=MW_LIMIT, debug=False):
  """
  Use reqMediaWiki to check if page titles are in a Wikimedia project and
  returns the Wikidata entity for them. Automatically resolves redirects. If a
  title is invalid (contains forbidden characters) in the Wikimedia project,
  the "status" column of the returned Pandas data-frame is set to "invalid." If
  the title does not exist, that column is set to "missing". If the page is a
  disambiguation page, then "disambiguation" is set in that column. If the page
  title exists (and not it is a disambiguation page), then 'OK' is set in that
  column. Note that additional columns are "normalized" and "target". The
  "normalized" column contains the normalized title of the page, if it is
  different from the original title, else None. The "target" columns contains
  the title of the target page if any, else None. For a title, if the status
  column is disambiguation or OK, then the column "entity" constains the
  Wikidata entity for the page.

  Note that duplicate titles are removed before searches. Note also that for
  two different titles with the same normalized or target title, the same
  information is returned for both.

  See https://www.mediawiki.org/w/api.php?action=help&modules=query%2Bpageprops

  :param titles: A title or a list of titles to search for.
  :param project: Wikimedia project, defaults "en.wikipedia.org"
  :param chunksize: If the number of titles exceed chunksize (the limit is 50
         for the MediaWiki API), then query are made in chunks if chunsize.
  :param debug: For debugging information.
  :return A Pandas data-frame with columns "title", "status", "nomalized", "target"
          and "entity".
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :examples:
  >>> m_WikidataEntity('Max Planck', project='es.wikipedia.org')
             status normalized redirect entity
  Max Planck     OK       None     None  Q9021

  >>> m_WikidataEntity(['Max', 'Cervante', 'humanist'])
                    status normalized    target   entity
  Max       disambiguation       None      None  Q225238
  Cervante         missing       None      None     None
  humanist              OK   Humanist  Humanism   Q46158

  >>> # Note that "a%CC%8C" is an "a" + combining caron (the diacritic mark: ˇ)
  >>> a = requests.utils.unquote("a%CC%8C")
  >>> m_WikidataEntity(['Max Planck', a, 'Max', 'Cervante', 'humanist'])
                      status normalized    target   entity
  Max Planck              OK       None      None    Q9021
  ǎ                      OK          Ǎ     Caron   Q26948
  Max         disambiguation       None      None  Q225238
  Cervante           missing       None      None     None
  humanist                OK   Humanist  Humanism   Q46158
  >>> # Search 3D films:
  ... w = w_SearchByInstanceof(instanceof='Q229390', langsorder = 'en', debug = 'info')
  ... films = list(w.entityLabel.values)
  ... m = m_WikidataEntity(films[:125], debug=True)
  """
  # Checking titles
  titles = checkTitles(titles)
  #
  # Chunked requests?
  n = len(titles)
  if n>chunksize:
    if debug!=False:
      print(f"INFO: The number of titles ({n}) exceeds the MediaWiki API limit ({chunksize}): doing chunked requests.", file=sys.stderr)
    return doChunks(m_WikidataEntity, titles, chunksize=MW_LIMIT, project=project, debug=debug)
  ###
  query = {"format"        : 'json',
           "formatversion" : '2',
           "redirects"     : '1',      # Automatically resolve redirects in query+titles, etc.
           "action"        : 'query',
           "prop"          : 'pageprops',
           "ppprop"        : 'wikibase_item|disambiguation',
           "titles"        : '|'.join(titles)}
  #
  output = dict()
  #
  # Note that for query "pageprops" continue responses are not expected
  # to happen when only request wikibase_item and disambiguation, but...
  while True:
    j = reqMediaWiki(query=query, project=project, debug=debug)

    if j is None or "query" not in j:
      return None
    #
    q = j['query']
    for title in titles:
      normalized, target = normalizedTitle(title, q)
      # Get the firts not None element:
      anorm = next(n for n in [target, normalized, title] if n is not None)
      #
      for page in q['pages']:      # With formatversion=2 q["pages"] is a list
        if anorm == page['title']:
          item = None
          if 'invalid' in page:      # Invalid response: ¿malformed title?
            status = "invalid"
          elif 'missing' in page:
            status = "missing"
          elif 'pageprops' not in page:    # No pageprops, so no wikidata_item
            status = "no_pageprops"
          elif 'wikibase_item' not in page['pageprops']:   # No wikibase_item
            status = "no_wikibase_item"
          else:
            status = "disambiguation" if 'disambiguation' in page['pageprops'] else 'OK'
            item = page['pageprops']['wikibase_item']
          #
          output[title] = {# 'title'     : title,
                           'status'    : status,
                           'normalized': normalized,
                           'target'    : target,
                           'entity'    : item}
    # ¿"continue" response? Only query 2 properties, it is certain that a
    # continue response will not happen, but...
    if 'continue' in j:
      print("INFO: continue response.", file=sys.stderr)
      query.update(j['continue'])
    else:
      break
  # Finally return de dict output as a Pantad data-frame.
  return pd.DataFrame.from_dict(output, orient='index')


#%% m_Redirects(titles, project="en.wikipedia.org", chunksize=MW_LIMIT, debug=False)
def m_Redirects(titles, project="en.wikipedia.org", chunksize=MW_LIMIT, debug=False):
  """
  Obtain the redirection pages to the article titles in the Wikimedia project,
  restricted to namespace 0. Return a dictionary, each key is a title and the
  value are a list of redirections. In this list the first elemnent is the
  target of the page title requested and contains all page source, each
  redirects to it. The first page is also normalized. If the title requested
  is invalid or missing in the project, then the value of the dict for it it
  set to None.

  :param titles: A title or a list of titles to redirects search for.
  :param project: The Wikimedia project, defaults "en.wikipedia.org"
  :param chunksize: If the number of titles exceed chunksize (the limit is 50
         for the MediaWiki API), then query are made in chunks if chunsize.
  :return A dict.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :examples:
  >>> m_Redirects(['Max', 'Eustaquio Celada', 'humanist', 'Cervante'])
  """
  # Checking titles
  titles = checkTitles(titles)
  #
  # Chunked requests?
  n = len(titles)
  if n>chunksize:
    if debug!=False:
      print(f"INFO: The number of titles ({n}) exceeds the MediaWiki API limit ({chunksize}): doing chunked requests.", file=sys.stderr)
    return doChunks(m_Redirects, titles, chunksize, project=project, debug=debug)
  ###
  query = {
      "format"       : "json",
      "formatversion": "2",
      "redirects"    : "1",        # Automatically resolve redirects in query+titles, etc.
      "action"       : "query",
      "prop"         : "redirects",
      "rdnamespace"  : "0",        # Only from namespace = 0
      "rdprop"       : "title",
      "rdlimit"      : "max",      # Change to 5 for testing
      "titles"       : '|'.join(titles),
  }
  #
  output = dict()
  #
  while True:
    j = reqMediaWiki(query=query, project=project, debug=debug)
    #
    if j is None or "query" not in j:
      return None
    #
    q = j['query']
    for title in titles:
      normalized, target = normalizedTitle(title, q)
      # Get the firts not None element:
      anorm = next(n for n in [target, normalized, title] if n is not None)
      #
      for page in q['pages']:      # With formatversion=2 q["pages"] is a list
        if anorm == page['title']:
          # Any continue response includes the same "normalized" and "redirects" info
          # When query is requested in the firts time, no "continue" key exists
          # in the "query" dict; only it is possible in succesive continue request.
          if 'continue' not in query:
            if 'invalid' in page or 'missing' in page:
              output[title] = [] # None
              break
            else:
              output[title] = [anorm]
          # But redirects can be different in continue responses:
          if 'redirects' in page:   #  redirected titles
            redirects = [r["title"] for r in page["redirects"]]
          else:
            redirects = []
          #
          output[title].extend(redirects)

    # continue response
    if 'continue' in j:
      print("INFO: continue response.", file=sys.stderr)
      query.update(j['continue'])
    else:
      break
  # Check None
  for k,v in output.items():
    if len(v) == 0:
      output[k] = None
  # Finally return a dict
  return output

#%% m_RedirectsDF(titles, project="en.wikipedia.org", chunksize=MW_LIMIT, debug=False)
def m_RedirectsDF(titles, project="en.wikipedia.org", chunksize=MW_LIMIT, debug=False):
  """
  Obtain the redirection pages to the article titles in the Wikimedia project,
  restricted to namespace 0. Return a Pandas data-frame with columns "status",
  "normalized", "target" and "redirects". The "status" column is set to
  "invalid", "missing" or "disambiguatoin" if the corresponding title is
  invalid, not exists or is a disambiguaton page, else "OK". The "normalize"
  columns contains the normalized title if any. The "redirect" column contains
  None if title is invalid or missing, else a list of redirects titles of the
  page. Note that all redirected title to the target is included in this list.
  The firts element in the list is the target title of the rest of the pages in
  the list.

  :param titles: A title or a list of titles to redirects search for.
  :param project: The Wikimedia project, defaults "en.wikipedia.org"
  :param chunksize: If the number of titles exceed chunksize (the limit is 50
         for the MediaWiki API), then query are made in chunks if chunsize.
  :return A dict.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :example:
  >>> m_Redirects(['Max', 'Eustaquio Celada', 'humanist', 'Cervante'], chunksize=4)
  """
  # Checking titles
  titles = checkTitles(titles)
  #
  # Chunked requests?
  n = len(titles)
  if n>chunksize:
    if debug!=False:
      print(f"INFO: The number of titles ({n}) exceeds the MediaWiki API limit ({chunksize}): doing chunked requests.", file=sys.stderr)
    return doChunks(m_RedirectsDF, titles, chunksize, project=project, debug=debug)
  ###
  query = {
      "format"       : "json",
      "formatversion": "2",
      "redirects"    : "1",        # Automatically resolve redirects in query+titles, etc.
      "action"       : "query",
      "prop"         : "redirects|pageprops",
      "rdnamespace"  : "0",        # Only from namespace = 0
      "rdprop"       : "title",
      "rdlimit"      : "max",      # Change to 5 for testing
      "ppprop"       : "wikibase_item|disambiguation",
      "titles"       : '|'.join(titles),
  }
  #
  output = dict()
  #
  while True:
    j = reqMediaWiki(query=query, project=project, debug=debug)
    #
    if j is None or "query" not in j:
      return None
    #
    q = j['query']
    for title in titles:
      output[title] = {'title': title}
      #
      normalized, target = normalizedTitle(title, q)
      # Get the firts not None element:
      anorm = next(n for n in [target, normalized, title] if n is not None)
      #
      for page in q['pages']:      # With formatversion=2 q["pages"] is a list
        if anorm == page['title']:
          item = None
          # Any continue response includes the same "normalized" and "redirects" info
          # When query is requested in the firts time, no "continue" key exists
          # in the "query" dict; only it is possible in succesive continue request.
          if 'continue' not in query:
            if 'invalid' in page:      # Invalid response: ¿malformed title?
              status = "invalid"
            elif 'missing' in page:
              status = "missing"
            elif 'pageprops' not in page:    # No pageprops, so no wikidata_item
              status = "no_pageprops"
            elif 'wikibase_item' not in page['pageprops']:   # No wikibase_item
              status = "no_wikibase_item"
            else:
              status = "disambiguation" if 'disambiguation' in page['pageprops'] else 'OK'
              item = page['pageprops']['wikibase_item']

            output[title].update({'status'    : status,
                                  'normalized': normalized,
                                  'target'    : target,
                                  'entity'    : item})
            #
          output[title].update({'redirects' : [] if item is None else [anorm]})
          if 'redirects' in page:   #  redirected titles
            redirects = [r["title"] for r in page["redirects"]]
          else:
            redirects = []
          #
          output[title]['redirects'].extend(redirects)
    # continue response
    if 'continue' in j:
      print("INFO: continue response.", file=sys.stderr)
      query.update(j['continue'])
    else:
      break
  # Check None
  for k,v in output.items():
    if len(v['redirects']) == 0:
      v['redirects'] = None
  # Finally return a DF
  return pd.DataFrame.from_dict(output, orient='index')
  return output


#%% m_PagePrimaryImage(titles, project='en.wikipedia.org', chunksize=MW_LIMIT, debug=False)
def m_PagePrimaryImage(titles, project='en.wikipedia.org', chunksize=MW_LIMIT, debug=False):
  """
  Use reqMediaWiki to return the URL of the image associated with the
  Wikipedia pages, if any. Automatically resolves redirects. If a
  title is invalid (contains forbidden characters) in the Wikimedia project,
  the "status" column of the returned Pandas data-frame is set to "invalid." If
  the title does not exist, that column is set to "missing". If the page title
  exists, then  'OK' is set in that column. Note that additional columns are
  "normalized" and "target". The "normalized" column contains the normalized
  title of the page, if it is different from the original title, else None.
  The "target" columns contains the title of the target page if any, else None.
  For a title, if the status column is OK, then the column "image" constains
  the image URL of the page, if any, else None. Note that the URL returned
  by the API is percent-escaped.

  Note that exists a action=query with prop=pageprop and ppprop="page_image_free"
  which also returns the primary image of the page, but not complete URL is
  returned with it. We want to know the complete URL for downloading de image
  in the future, so the funtion uses prop=pageimages.

  Note that duplicate titles are removed before searches. Note also that for
  two different titles with the same normalized or target title, the same
  information is returned for both.

  See https://www.mediawiki.org/w/api.php?action=help&modules=query%2Bpageimages

  :param titles: A title or a list of titles to search for.
  :param project: Wikimedia project, defaults "en.wikipedia.org"
  :param chunksize: If the number of titles exceed chunksize (the limit is 50
         for the MediaWiki API), then query are made in chunks if chunsize.
  :return A Pandas data-frame with columns "status", "nomalized", "target"
          and "image".
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :examples:
  >>> m_PagePrimaryImage('Max Planck', project='es.wikipedia.org')
             status normalized redirect image
  Max Planck     OK       None     None https://upload.wikimedia.org/wikipedia/commons...

  >>> m_PagePrimaryImage(['Max', 'Cervante', 'humanist'])
             status normalized    target  image
  Max            OK       None      None  None
  Cervante  missing       None      None  None
  humanist       OK   Humanist  Humanism  https://upload.wikimedia.org/wikipedia/commons...

  >>> # Note that "a%CC%8C" is an "a" + combining caron (the diacritic mark: ˇ)
  >>> a = requests.utils.unquote("a%CC%8C")
  >>> m_PagePrimaryImage(['Max Planck', a, 'Max', 'Cervante', 'humanist'])
               status normalized    target  image
  Max Planck       OK       None      None  https://upload.wikimedia.org/wikipedia/commons...
  ǎ               OK          Ǎ     Caron  None
  Max              Ok       None      None  None
  Cervante    missing       None      None  None
  humanist         OK   Humanist  Humanism  https://upload.wikimedia.org/wikipedia/commons...
  >>> # Search 3D films, get Wikipedia pages in English and get primary images.:
  ... w = w_SearchByInstanceof(instanceof='Q229390', langsorder = 'en', debug = 'info')
  ... wk = w_Wikipedias(w.entity, wikilangs='en')
  ... films = wk.names[wk.names!='']
  ... m = m_PagePrimaryImage(films, debug='info')
  """
  # Checking titles
  titles = checkTitles(titles)
  #
  # Chunked requests?
  n = len(titles)
  if n>chunksize:
    print(f"INFO: The number of titles ({n}) exceeds the MediaWiki API limit ({MW_LIMIT}): doing chunked requests.", file=sys.stderr)
    return doChunks(m_PagePrimaryImage, titles, chunksize, project=project, debug=debug)
  ###
  query = {"format"        : 'json',
           "formatversion" : '2',
           "redirects"     : '1',          # Automatically resolve redirects in query+titles, etc.
           "action"        : 'query',
           "prop"          : 'pageimages',
           "piprop"        : 'original',
           "pilimit"       : "max",        # Change to 2 to test
           "titles"        : '|'.join(titles)}
  #
  output = dict()
  #
  # Note that for query "pageimages" continue responses are not expected
  # to happen (only exists one image URL per page, if any)
  j = reqMediaWiki(query=query, project=project, debug=debug)
  #
  if j is None or "query" not in j:
    return None
  #
  q = j['query']
  for title in titles:
    normalized, target = normalizedTitle(title, q)
    # Get the firts not None element:
    anorm = next(n for n in [target, normalized, title] if n is not None)
    #
    for page in q['pages']:      # With formatversion=2 q["pages"] is a list
      if anorm == page['title']:
        image = None
        if 'invalid' in page:      # Invalid response: ¿malformed title?
          status = "invalid"
        elif 'missing' in page:
          status = "missing"
        elif 'pageid' not in page:
          status = 'no pageid'
        else:
          status = 'OK'
          # image = ''
          if 'original' in page and 'source' in page['original']:
            image = page['original']['source']
        #
        output[title] = {'status'    : status,
                         'normalized': normalized,
                         'target'    : target,
                         'image'     : image}
  # Finally return de dict output as a Pantad data-frame.
  return pd.DataFrame.from_dict(output, orient='index')


#%% m_PageFiles(titles, project='en.wikipedia.org', chunksize=MW_LIMIT, exclude_ext='svg,webp,xcf', debug=False)
def m_PageFiles(titles, project='en.wikipedia.org', chunksize=MW_LIMIT, exclude_ext='svg,webp,xcf', debug=False):
  """
  Search for all URL files in the Wikipedia pages, usually image files. Exclude
  files with extensions in exclude_ext parameter. Also exclude files without
  extension.

  See https://en.wikipedia.org/w/api.php?action=help&modules=query%2Bimages

  Note that this query returns the name of the files, not the URL of the files.
  If you need the URL for downloding it is necessary to make a search for
  imageinfo (see the function m_ImageURL()). Also note that the name of the
  image is not percent-encoded, because it is not a URL. Also note that a
  generator (see example) return all images, but not references which images
  belongss to each title.
  https://en.wikipedia.org/w/api.php?format=json&formatversion=2&redirects=1&action=query&generator=images&gimlimit=max&prop=imageinfo&iiprop=url&titles=Cervantes

  :param titles: A title or a list of titles to search for.
  :param project: Wikimedia project, defaults "en.wikipedia.org"
  :param exclude_ext: Extensions of file to exclude in results. Default
         'svg,webp,xcf'.
  :return A Pandas data-frame with four columns. The "status" column is set to
          "invalid", "missing" of "OK" if title is not valid, does not exists
          o is valid, respectively. The "normalized" and "target" columns
          contains the normalized and the page destiny if the original title
          has any. The column "files" contains the files in the page (excluding
          ones have extensions in the "exclude_ext" paramenter), as a Python
          list.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :examples:
  >>> # Search 3D films:
  >>> # Search 3D films, get Wikipedia pages in English and get primary images.:
  ... w = w_SearchByInstanceof(instanceof='Q229390', langsorder = 'en', debug = 'info')
  ... wk = w_Wikipedias(w.entity, wikilangs='en')
  ... films = wk.names[wk.names!='']
  ... m = m_PageFiles(films, debug='info')
  """
  # Checking titles
  titles = checkTitles(titles)
  #
  # Chunked requests?
  n = len(titles)
  if n>chunksize:
    if debug!=False:
      print(f"INFO: The number of titles ({n}) exceeds the MediaWiki API limit ({MW_LIMIT}): doing chunked requests.", file=sys.stderr)
    return doChunks(m_PageFiles, titles, chunksize, project=project,
                       exclude_ext=exclude_ext, debug=debug)
  #
  # Extensions to exclude (ignoring case)
  exts = [x.lower() for x in re.split('\W+', exclude_ext)]
  ###
  query = {"format"        : 'json',
           "formatversion" : '2',
           "redirects"     : '1',       # Automatically resolve redirects in query+titles, etc.
           "action"        : 'query',
           "prop"          : 'images',
           "imlimit"       : 'max',
           "titles"        : '|'.join(titles)}
  #
  output = dict()
  #
  while(True):  # While there are "continue" responses
    j = reqMediaWiki(query=query, project=project, debug=debug)
    #
    q = j['query']
    for title in titles:
      normalized, target = normalizedTitle(title, q)
      # Get the firts not None element:
      anorm = next(n for n in [target, normalized, title] if n is not None)
      #
      for page in q['pages']:      # With formatversion=2 q["pages"] is a list
        if anorm == page['title']:
          # Any continue response includes the same "normalized" and "redirects" info
          # When query is requested in the firts time, no "continue" key exists
          # in the "query" dict; only it is possible in succesive continue request.
          if 'continue' not in query:
            if 'invalid' in page:      # Invalid response: ¿malformed title?
              status = "invalid"
            elif 'missing' in page:
              status = "missing"
            elif 'pageid' not in page:
              status = 'no pageid'
            else:
              status = 'OK'
            #
            output[title] = {'status'    : status,
                             'normalized': normalized,
                             'target'    : target,
                             'nfiles'    : 0,
                             'files'     : None if status != 'OK' else []}
            #
          if 'images' in page:
            images = list()
            for image in page['images']:
              img = image['title']
              fn, ext = splitext(img)   # ext include the dot ".ext"
              if ext=='' or ext[1:] in exts:
                continue
              images.append(image['title'])
            output[title]['nfiles'] += len(images)
            output[title]['files'].extend(images)
    #
    # continue response
    if 'continue' in j:
      print("INFO: continue response.", file=sys.stderr)
      query.update(j['continue'])
    else:
      break
  # Finally return de dict output as a Pantad data-frame.
  return pd.DataFrame.from_dict(output, orient='index')

#%% m_ImageURL(titles, project='en.wikipedia.org', chunksize=MW_LIMIT, debug=False)
def m_ImageURL(titles, project='en.wikipedia.org', chunksize=MW_LIMIT, debug=False):
  """
  Return the URL of the titles (titles in the File namespace, in which all of
  Wikipedia's media content resides) in the Wikipedia project. Returns the URL
  of the Wikimedia Commons space. Note that the titles must be prefixed with
  a "file" prefix: "File:" in the English Wikipedia, "Archivo:" in the Spanish
  Wikipedia and so on.

  See https://www.mediawiki.org/w/api.php?action=help&modules=query%2Bimageinfo

  :param titles: A titles or a list of titles to search for (prefixed with "File:").
  :param project: Wikimedia project, defaults "en.wikipedia.org"
  :return A dict with titles (file images) and the associated URL. The URL is
          percent-escaped as returned by the API.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :examples:
  >>> # Search 3D films
  ... w = w_SearchByInstanceof(instanceof='Q229390', langsorder = 'en', debug = 'info')
  ... films = list(w.entityLabel.values)
  ... m = m_PageFiles(films, debug='info')
  ... filenames = [f for l in m.files.values for f in l]
  ... urls = m_ImageURL(filenames[:350], debug='info')
  """
  # Checking titles
  titles = checkTitles(titles)
  #
  # Chunked requests?
  n = len(titles)
  if n>chunksize:
    if debug!=False:
      print(f"INFO: The number of titles ({n}) exceeds the MediaWiki API limit ({MW_LIMIT}): doing chunked requests.", file=sys.stderr)
    return doChunks(m_ImageURL, titles, chunksize, project=project, debug=debug)
  ###
  query = {"format"        : 'json',
           "formatversion" : '2',
           "redirects"     : '1',       # Automatically resolve redirects in query+titles, etc.
           "action"        : 'query',
           "prop"          : 'imageinfo',
           "iiprop"        : "url",  #  "url|size",
           "iilimit"       : 'max',
           "titles"        : '|'.join(titles)}
  #
  output = dict()
  #
  # Note that for query "imageinfo" continue responses are not expected
  # to happen (only exists one image URL per image, if any)
  j = reqMediaWiki(query=query, project=project, debug=debug)
  #
  if j is None or "query" not in j:
    return None
  #
  q = j['query']
  for title in titles:
    normalized, target = normalizedTitle(title, q)
    # Get the firts not None element:
    anorm = next(n for n in [target, normalized, title] if n is not None)
    #
    for page in q['pages']:      # With formatversion=2 q["pages"] is a list
      if anorm == page['title']:
        if 'known' in page:      # Note that "known" appears next to "missing"
          status = 'OK'
        elif 'invalid' in page:      # Invalid response: ¿malformed title?
          status = "invalid"
        elif 'missing' in page:
          status = "missing"
        elif "filehidden" in page:
          status = "filehidden"
        else:
          status = 'OK'
        #
        output[title] = {'status'    : status,
                         'normalized': normalized,
                         'target'    : target,
                         # 'URL'       : None if status != 'OK' else ""
                         }
        #
        if 'imageinfo' in page: # formatversion2 returns a list with only one element
          output[title]['URL'] = page['imageinfo'][0]['url']
        else:
          output[title]['URL'] = None
  # Finally return de dict output as a Pantad data-frame.
  return pd.DataFrame.from_dict(output, orient='index')


#%% m_PageOutLinks(titles, project='en.wikipedia.org', chunksize=MW_LIMIT, debug=False)
def m_PageOutLinks(titles, project='en.wikipedia.org', chunksize=MW_LIMIT, debug=False):
  """
  Return for each page all outgoing links it has to other pages in the
  Wikimedia project. Only in namespace 0. Note that redirects is in effect,
  then the outgoing links from the target page are returned.

  See https://www.mediawiki.org/w/api.php?action=help&modules=query%2Blinks

  :param titles: A title or a list of titles to search for.
  :param project: Wikimedia project, defaults "en.wikipedia.org"

  :return A Pandas data-frame with four columns. The "status" column is set to
          "invalid", "missing" of "OK" if title is not valid, does not exists
          o is valid, respectively. The "normalized" and "target" columns
          contains the normalized and the page destiny if the original title
          has any. The column "links" contains the links to othes pages in the
          Wikimedia project in namespace 0.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :examples
  >>> # Search 3D films, get Wikipedia pages in English and get primary images.:
  ... w = w_SearchByInstanceof(instanceof='Q229390', langsorder = 'en', debug = 'info')
  ... wk = w_Wikipedias(w.entity, wikilangs='en')
  ... films = wk.names[wk.names!='']
  ... m = m_PageOutLinks(films[:25], debug='info')
  """
  # Checking titles
  titles = checkTitles(titles)
  #
  # Chunked requests?
  n = len(titles)
  if n>chunksize:
    if debug!=False:
      print(f"INFO: The number of titles ({n}) exceeds the MediaWiki API limit ({MW_LIMIT}): doing chunked requests.", file=sys.stderr)
    return doChunks(m_PageOutLinks, titles, chunksize, project=project, debug=debug)
  ###
  query = {"format"        : 'json',
           "formatversion" : '2',
           "redirects"     : '1',        # Automatically resolve redirects in query+titles, etc.
           "action"        : 'query',
           "prop"          : 'links',
           "plnamespace"   : '0',
           "pllimit"       : 'max',
           "titles"        : '|'.join(titles)}
  #
  output = dict()
  #
  while(True):  # While there are "continue" responses
    j = reqMediaWiki(query=query, project=project, debug=debug)
    #
    if j is None or "query" not in j:
      return None
    #
    q = j['query']
    for title in titles:
      normalized, target = normalizedTitle(title, q)
      # Get the firts not None element:
      anorm = next(n for n in [target, normalized, title] if n is not None)
      #
      for page in q['pages']:      # With formatversion=2 q["pages"] is a list
        if anorm == page['title']:
          # Any continue response includes the same "normalized" and "redirects" info
          # When query is requested in the firts time, no "continue" key exists
          # in the "query" dict; only it is possible in succesive continue request.
          if 'continue' not in query:
            if 'invalid' in page:      # Invalid response: ¿malformed title?
              status = "invalid"
            elif 'missing' in page:
              status = "missing"
            elif 'pageid' not in page:
              status = 'no pageid'
            else:
              status = 'OK'
            #
            output[title] = {'status'    : status,
                             'normalized': normalized,
                             'target'    : target,
                             'nlinks'    : 0,
                             'links'     : None if status != 'OK' else []}
            #
          if 'links' in page:
            links = [d['title'] for d in page['links']]
            output[title]['nlinks'] += len(links)
            output[title]['links'].extend(links)
    #
    # continue response
    if 'continue' in j:
      if debug!=False:
        print("  INFO: continue response.", file=sys.stderr)
      query.update(j['continue'])
    else:
      break
  # Finally return de dict output as a Pantad data-frame.
  return pd.DataFrame.from_dict(output, orient='index')

#%% m_PageInLinks(titles, project='en.wikipedia.org', redirects=True, chunksize=MW_LIMIT, debug=False)
def m_PageInLinks(titles, project='en.wikipedia.org', redirects=True, chunksize=MW_LIMIT, debug=False):
  """
  Return for each page all incoming links it has from other pages in the
  Wikimedia project (incoming_links). Only in namespace 0. Note that possible
  redirects to the page are not included. If redirects=True, then the function
  returns all incoming links to redirects and target page.

  See https://www.mediawiki.org/w/api.php?action=help&modules=query%2Blinkshere

  :param titles: A title or a list of titles to search for.
  :param project: Wikimedia project, defaults "en.wikipedia.org"

  :return A Pandas data-frame with four columns. The "status" column is set to
          "invalid", "missing" of "OK" if title is not valid, does not exists
          o is valid, respectively. The "normalized" and "target" columns
          contains the normalized and the page destiny if the original title
          has any. The column "linkshere" contains the in-links from other
          pages in the Wikimedia project in namespace=0
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :examples
  >>> # Search 3D films, get Wikipedia pages in English and get primary images.:
  ... w = w_SearchByInstanceof(instanceof='Q229390', langsorder = 'en', debug = 'info')
  ... wk = w_Wikipedias(w.entity, wikilangs='en')
  ... films = wk.names[wk.names!='']
  ... m = m_PageInLinks(films[:25], debug='info')
  """
  # Checking titles
  titles = checkTitles(titles)
  #
  if redirects:
    # Obtain all redirects of any target page in titles
    titlestarget = m_Redirects(titles, project)
    titles = [x for v in titlestarget.values() if v is not None for x in v ]  # Flatten the titlestarget
    # Adding titles invalid or missing (title: None)
    titles.extend([x for x in titlestarget.keys() if titlestarget[x] is None])
    # Remove duplicates preserving order
    titles = list(dict.fromkeys(titles))
  #
  # Chunked requests?
  n = len(titles)
  if debug!=False and redirects:
    print(f"INFO: Parameter 'redirects' is True: The number of redirects to all titles in list is: {n}.", file=sys.stderr)

  if n>chunksize:
    if debug!=False:
      print(f"INFO: The number of titles ({n}) exceeds the MediaWiki API limit ({MW_LIMIT}): doing chunked requests.", file=sys.stderr)
      timeinit = time()
      #
    output=dict()
    n = len(titles)
    nlim = int(n/chunksize)
    for k in range(nlim+1):
      offset = k*chunksize
      x_list = titles[offset:offset+chunksize]
      if len(x_list)==0:
        break
      if debug:
        t0=time()
        print(f" INFO: Executing the function on elements from {offset+1} to {offset+len(x_list)}", end="", file=sys.stderr)
      # Note that d is a dataframe
      d = m_PageInLinks(x_list, project=project, redirects=False, chunksize=chunksize, debug=debug)
      if debug:
        print(f" ({time()-t0:.2f} seconds)", file=sys.stderr)
      if (k==0):
        output = d
      else: # Note that d is a dataframe
        output = pd.concat([output, d])
    if debug:
      print(f" INFO: Total time {time()-timeinit:.2f} seconds", file=sys.stderr)
      #
    if not redirects:
      return output
      #
    # Redirects = True -> dataframe -> to dict -> to dataframe
    outputtarget = dict()
    for title,tredir in titlestarget.items():
      outputtarget[title] = output.loc[title].to_dict()
      if tredir is None:
        continue
      linkshere = []
      for t in tredir:
        linkshere.extend(output.loc[t]['linkshere'])
      # Remove duplicates preserving order
      linkshere = list(dict.fromkeys(linkshere))
      outputtarget[title]['nlinks']    = len(linkshere)
      outputtarget[title]['linkshere'] = linkshere
    return pd.DataFrame.from_dict(outputtarget, orient='index')

  # if not n>chunksize
  query = {"format"        : 'json',
           "formatversion" : '2',
           # "redirects"     : '1',
           "action"        : 'query',
           "prop"          : 'linkshere',
           "lhnamespace"   : '0',
           "lhprop"        : "title",
           # "lhshow"        : "!redirect",
           "lhlimit"       : 'max',
           "titles"        : '|'.join(titles)}
  #
  output = dict()
  #
  while(True):  # While there are "continue" responses
    j = reqMediaWiki(query=query, project=project, debug=debug)
    #
    if j is None or "query" not in j:
      return None
    #
    q = j['query']
    #
    for title in titles:
      normalized, target = normalizedTitle(title, q)
      # Get the firts not None element:
      anorm = next(n for n in [target, normalized, title] if n is not None)
      #
      for page in q['pages']:      # With formatversion=2 q["pages"] is a list
        if anorm == page['title']:
          # Any continue response includes the same "normalized" and "redirects" info
          # When query is requested in the firts time, no "continue" key exists
          # in the "query" dict; only it is possible in succesive continue request.
          if 'continue' not in query:
            if 'invalid' in page:      # Invalid response: ¿malformed title?
              status = "invalid"
            elif 'missing' in page:
              status = "missing"
            elif 'pageid' not in page:
              status = 'no pageid'
            else:
              status = 'OK'
            #
            output[title] = {'status'    : status,
                             'normalized': normalized,
                             'target'    : target,
                             'nlinks'    : 0,
                             'linkshere' : []}
          if 'linkshere' in page:
            links = [d['title'] for d in page['linkshere']]
            output[title]['nlinks'] += len(links)
            output[title]['linkshere'].extend(links)
    #
    # continue response
    if 'continue' in j:
      if debug!=False:
        print("  INFO: continue response.", file=sys.stderr)
      query.update(j['continue'])
    else:
      break

  # Finally return de dict output as a Panda data-frame.
  if not redirects:
    return pd.DataFrame.from_dict(output, orient='index')
  # Redirects = True
  outputtarget = dict()
  for title,tredir in titlestarget.items():
    outputtarget[title] = output[title]
    if tredir is None:
      continue
    linkshere = []
    for t in tredir:
      linkshere.extend(output[t]['linkshere'])
    # Remove duplicates preserving order
    linkshere = list(dict.fromkeys(linkshere))
    outputtarget[title]['nlinks']    = len(linkshere)
    outputtarget[title]['linkshere'] = linkshere
  return pd.DataFrame.from_dict(outputtarget, orient='index')


#%% -- WikiMedia REST API ------------------------------------------------------
#  The Wikimedia REST API provides e.g. pageviews and aggregate edit stats
#' See https://www.mediawiki.org/wiki/Wikimedia_REST_API
#' See https://www.mediawiki.org/wiki/XTools/API/Page (xtools.wmcloud.org)
#' See https://en.wikipedia.org/api/rest_v1/
#'
#' Note that this API uses "article" to refer a page title. We also use this
#' approach. In this API only one article is allowed in each request.

#%% m_PageViews(article, start, stop, project, access, agent, granularity)
def m_PageViews(article,           # the title of the article (without "_")
      start, end,                  # The data of first/last day to inclue (YYYYMMDD or YYYYMMDDHH)
      project = "en.wikipedia.org", # Filter by Wikimedia project
      access  = "all-access",       # Filter by access method: all-access, desktop, mobile-app, mobile-web
      agent   = "all-agents",       # Filter by agent type: all-agents, user, spider, automated
      granularity = "monthly",      # time unit for the response data: daily, monthly
      redirects = False,
      debug=False):
  """
  Use the Wikimedia REST API (https://wikimedia.org/api/rest_v1/) to get the
  number of views one article has in a Wikimedia project in a date interval
  (see granularity). If redirect=True, then get the number of views of all
  articles that redirects to the article which is the target of the original
  page.

  The API has a rate limit of 100 req/s, but with a sequential search the speed
  limit is never reached due to network latency.
  [See https://wikimedia.org/api/rest_v1/#/Pageviews%20data]

  :param article: The title of the article to search. Only one article is allowed.
  :param start,end: First and last day to include (format YYYYMMDD or YYYYMMDDHH)
  :param project: The Wikimedia project, defaults en.wikipedia.org
  :param access:  Filter by access method: all-access (default), desktop, mobile-app, mobile-web
  :param agent:   Filter by agent type: all-agents, user (default), spider, automated
  :param granularity:  Time unit for the response data: daily, monthly (default)
  :param redirects: Boolean for including the views of all redirections of the
         page (defaults: False). If redirects=True then the "normalized" column
         element of the returned data-frame contains the target of the
         redirection, and the "index" element contains the original title of
         the article.  If a page is just a target of other pages, and you want
         to know the total number of views that page have (including views of
         redirections), it is also necessary set redirects=True, otherwise only
         you have the views of that page.
  :param debug: For debugging purpouses.
  :return A Counter with the number of views by granularity.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  """
  if article.strip() == '':
    return None

  if redirects:
    r = m_Redirects(article, project)
    articles = r[article]['redirects']
  else:
    articles = [article]

  views = Counter()
  url = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
  for art in articles:
    art = art.replace(" ", "_")
    url += f"{project}/{access}/{agent}/{art}/{granularity}/{start}/{end}"
    response = requests.get(url=url, headers={'user-agent': user_agent})
    response.raise_for_status()
    if debug:
      print(requests.utils.unquote(response.url), file=sys.stderr)
    j = response.json()
    #
    if 'items' in j:
      views.update(Counter({item['timestamp']:item['views'] for item in j['items']}))
  #
  return views

#%% m_PageInfoType(article, infotype="articleinfo", project="en.wikipedia.org", redirects=True
def m_PageInfoType(article, infotype="articleinfo", project="en.wikipedia.org",
                   redirects=True, debug=False):
  """
  Obtain information in JSON format about an article in the Wikimedia project
  or None on errors. Uses the wmflabs API. The XTools Page API endpoints offer
  data related to a single page.

   See https://www.mediawiki.org/wiki/XTools/API/Page [xtools.wmflabs.org].
   The URL of the API starts with 'https://xtools.wmcloud.org/api/page/'

  :param article: The title of the article to search. Only one article is allowed.
  :param project: The Wikimedia project, defaults en.wikipedia.org.
  :param infotype: The type of information to request: articleinfo, prose, links
     articleinfo: Get basic information about the history of a page.
     prose: Get statistics about the prose (characters, word count, etc.) and
            referencing of a page.
     links: Get the number of incoming and outgoing links and redirects to the
            given page.
     Note that the API also offer theses options: top_editors, assessments,
     bot_data and automated_edits.

  :param redirects: If redirects=TRUE, then the information is obtained
         from the destiny of the page. In that case, the "original" element
         of the returned list contains the original page, and the "page"
         element the target page. Also, if infotype='links', the sum of the
         in-links of all redirections is assigned to links_in_count.

   NOTE: With "articleinfo" and "links" options the API gives information
       about the page itself, not about the possible page to which it redirects
       (a target page). However, with the "prose" option, information is
       provided on the target page.
  :return A dict with the information about the page.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  """
  #
  if infotype == 'prose':
    # Redirect is ignored
    articles = [article]
  elif redirects:
    r = m_Redirects(article, project)
    articles = r[article]['redirects']
    # r = m_Redirects2(article, project)
    # articles = r[article]
    if infotype == 'articleinfo':
      articles = articles[0:1]
  else:
    articles = [article]

  for i in range(len(articles)):
    art = articles[i]
    art = art.replace(" ", "_")
    url = f"https://xtools.wmflabs.org/api/page/{infotype}/{project}/{art}"
    response = requests.get(url=url, headers={'user-agent': user_agent})
    response.raise_for_status()
    if debug:
      print(requests.utils.unquote(response.url), file=sys.stderr)
    j = response.json()
    #
    if i==0:
      d = j
    else:    # only i>0 if infotype == 'links':
      d['links_out_count'] += j['links_out_count']
      d['links_ext_count'] += j['links_ext_count']
      d['links_in_count']  += j['links_in_count']
      d['redirects_count'] += j['redirects_count']
      d['elapsed_time']    += j['elapsed_time']
    #
  return d


#%% m_PageInfo(article, project="en.wikipedia.org", redirects=True)
def m_PageInfo(article, project="en.wikipedia.org", redirects=True, debug=False):
  """
  Obtain information in JSON format about an article in the Wikimedia project
  or None on errors. Uses the wmflabs API. The XTools Page API endpoints offer
  data related to a single page. The dict information contains information
  about 'articleinfo', 'prose' and 'links'. Note that the API also offer
  theses options: top_editors, assessments, bot_data and automated_edits. The
  information returned with each options is:

     articleinfo: Get basic information about the history of a page.

     prose: Get statistics about the prose (characters, word count, etc.) and
            referencing of a page.

     links: Get the number of incoming and outgoing links and redirects to the
            given page.

   IMPORTANT: With "articleinfo" and "links" options the API gives information
       about the page itself, not about the possible page to which it redirects
       (a target page). However, with the "prose" option, information is
       provided on the target page.

   See https://www.mediawiki.org/wiki/XTools/API/Page [xtools.wmflabs.org].
   The URL of the API starts with 'https://xtools.wmcloud.org/api/page/'

  :param article: The title of the article to search. Only one article is allowed.
  :param project: The Wikimedia project, defaults en.wikipedia.org.

  :param redirects: If redirects=True, then the information is obtained
         from the destiny of the page. In that case, for infotype='links, the
         sum of the in-links of all redirections is assigned to links_in_count.
  :return A dict with the information about the page.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  """
  output = dict()
  for infotype in ['articleinfo', 'prose', 'links']:
    #
    if infotype == 'prose':
      # Redirect is ignored
      articles = [article]
    elif redirects:
      r = m_Redirects(article, project)
      articles = r[article]['redirects']
      # r = m_Redirects2(article, project)
      # articles = r[article]
      if infotype == 'articleinfo':
        articles = articles[0:1]
    else:
      articles = [article]

    for i in range(len(articles)):
      art = articles[i]
      art = art.replace(" ", "_")
      url = f"https://xtools.wmflabs.org/api/page/{infotype}/{project}/{art}"
      response = requests.get(url=url, headers={'user-agent': user_agent})
      response.raise_for_status()
      if debug:
        print(requests.utils.unquote(response.url), file=sys.stderr)
      j = response.json()
      #
      if i==0:
        d = j
      else:    # only i>0 if infotype == 'links':
        d['links_out_count'] += j['links_out_count']
        d['links_ext_count'] += j['links_ext_count']
        d['links_in_count']  += j['links_in_count']
        d['redirects_count'] += j['redirects_count']
        d['elapsed_time']    += j['elapsed_time']

    output.update(d)
  #
  del output['elapsed_time']
  return output


#%% -- BNE, GETTY ----------------------------------------------------------
#
# Uses SPARQL endpoint or HTTP to search in BNE (Biblioteca Nacional de España)
# and Getty
# The SPARQL endpoint have a erratic behavior.

### Get the TTL triplet file (as string) from BNE identifier
def b_GetTTL(bneid):
  """
  Retrieve the TTL triplet file (as string) from BNE identifier. Use a simple
  URL request.
  :param benid: The identifier of a entity in BNE (XXxxxxxxx)
  :return A string with the TTL file.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  """
  if bneid == '':
    return None
  #
  bneid = bneid.upper()
  url = "https://datos.bne.es/persona/" + bneid + ".ttl"
  response = requests.get(url=url, headers={'user-agent': user_agent})
  response.raise_for_status()
  return response.text

### Get gender from a TTL triplet (as string)
def b_GenderTTL(bnettl):
  """
  Get gender or sex from a TTL triplet, if any, else "".

  :param bneTTL: A string with the TTL file content.
  :return A string with gender (in Spanish)
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  """
  m = re.search(r'@prefix\s+ns(\d+):\s+<http://www\.rdaregistry\.info/Elements/a/>', bnettl)
  if m is not None:
    ns = m.group(1)
  else:
    return ""
  m = re.search(ns+':P50116\s"([^"]+)', bnettl)
  if m is None:
    return ""
  return m.group(1)

### Search by Label (exact) usign the BNE Sparql endpoint.
def b_SearchByLabel(name, debug=False):
  """
  Use the SPARQL endpoint of datos.bne.es to search by label (exact search).
  Return a Pandas dataframe or None.
  :param name: Name to search (exact search)
  :return A data-frame with label, gender, birthdate, deathdate, occupations,
          and titles of works.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :examples:
  >>> b = b_SearchByLabel('Escobar, Modesto')
  """
  query = f"""prefix ns1: <https://datos.bne.es/resource>
prefix ns2: <https://datos.bne.es/def/>
prefix ns4: <http://www.rdaregistry.info/Elements/a/>
SELECT DISTINCT ?entity ?label ?genero ?fnac ?fmor
 (GROUP_CONCAT(DISTINCT ?oc;separator="\\n") as ?ocs)
 (GROUP_CONCAT(DISTINCT ?title;separator="\\n") as ?titles)
WHERE {{
  ?entity rdfs:label "{name}" .
  ?entity rdf:type  ns2:C1005 .
  OPTIONAL {{?entity ns2:P5001 ?label}}
  OPTIONAL {{?entity ns4:P50116 ?genero}}
  OPTIONAL {{?entity ns2:P5010 ?fnac}}
  OPTIONAL {{?entity ns2:P5011 ?fmor}}
  OPTIONAL {{?entity ns4:P50104 ?oc}}
  OPTIONAL {{?bimo   ns2:OP3006|ns2:OP1001|ns2:OP3003 ?entity.
             ?bimo   ns2:P3002|ns2:P1001  ?title.
           }}
}} GROUP BY ?entity ?label ?genero ?fnac ?fmor
"""
  url = 'https://datos.bne.es/sparql'
  params = {'format': 'json',  # 'format': 'application/sparql-results+json',
            'query': query}
  if debug:
    print(query, file=sys.stderr)
  #
  response = requests.get(url=url, params=params) #, headers=headers)
  response.raise_for_status()
  j = response.json()
  #
  bindings = j['results']['bindings']
  if len(bindings) == 0:
    if debug:
      print('No bindings')
    return None
    #
  data = list()
  for b in bindings:  # cada b es un dict
    d = {var:""  for var in j['head']['vars']}
    for k in b:  # Los campos devuelto están en j['head']['vars']
      d[k] = b[k]['value']
      if k in ['occ', 'titles']:
        d[k] = d[k].split("\n")
    data.append(d)
    #
  # return d
  return pd.DataFrame.from_dict(data)

### Use the BNE Sparql endpoint to retrieve the gender of the records which
### identifiers are in BNE_list
def b_Gender(BNE_list, chunksize=1500, debug=False):
  """
  Use the BNE Sparql endpoint to retrieve the gender of the records which
  identifiers ar in BNE_list.

  :param SUDOC_list: A list with SUDOC identifiers
  :param chunksize:
   We think that the SPARQL Query API has a limit of 60 seconds to return any
   request. The larger the number of entities in the query, the higher the risk
   of reaching this limit, so it is necessary to make several requests with a
   reduced number of entities. The parameter chunksize sets the maximum number
   of entities to be sent in each query. If 500 requests error is returned, it
   is necessary to decreace the chunksize value.
  :return A Pandas dataframe.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  """
  if isinstance(BNE_list, str):
    BNE_list = [BNE_list]
  #
  n = len(BNE_list)
  # Number of entities exceeds chunksize:
  if n > chunksize:
    print(f"INFO: The number of entities ({n}) exceeds chunksize ({chunksize}).", sep="", file=sys.stderr)
    for k in range(int(n/chunksize)+1):
      offset = chunksize*k
      q_list = BNE_list[offset:offset+chunksize]
      if len(q_list) == 0:
        break
      print(f"\tINFO: Requesting entities from {offset+1} to {offset+len(q_list)}.", file=sys.stderr)
      d = b_Gender(q_list, chunksize=chunksize, debug=debug)
      if k==0:
        output = d
      else:
        output.update(d)
    return output
  #
  values = "ns1:" + " ns1:".join(BNE_list)
  #
  url = 'https://datos.bne.es/sparql'
  #
  query = f"""prefix ns1: <https://datos.bne.es/resource/>
prefix ns4: <http://www.rdaregistry.info/Elements/a/>
SELECT DISTINCT ?bne ?label
(GROUP_CONCAT(DISTINCT ?sex;separator="|") as ?gender)
WHERE {{
  VALUES ?bne {{ {values} }}
  OPTIONAL {{?bne rdfs:label ?label.}}
  OPTIONAL {{?bne ns4:P50116 ?sex.}}
}} GROUP BY ?bne ?label
"""
  if debug:
    print(query, file=sys.stderr)
  #
  params = {'format': 'json',
            'query': query}
  response = requests.post(url=url, data=params, headers={'user-agent': user_agent})
  response.raise_for_status()
  j = response.json()
  bindings = j['results']['bindings']
  if len(bindings) == 0:
    return None
  #
  d = dict()
  for b in bindings:  # cada b es un dict
    bne = b['bne']['value'].replace('https://datos.bne.es/resource/','')
    d[bne] = {}
    for k in ['label', 'gender']:
      d[bne][k] = ''
      if k in b:
        d[bne][k] = b[k]['value']
  #return d
  # Return a Pandas dataframe
  return pd.DataFrame.from_dict(d, orient='index')

### Scrapping: send a HTTP request to search in catalogo.bne.es for extract
### the gender of a list of BNE identifiers. For each indetifier a complete
### MARC record (in string format) is returned.
def b_GenderScrapping(BNE_list):
  """
  Creo que habrá que cortar en unos 500 cada vez.
  """
  if isinstance(BNE_list, str):
    BNE_list = [BNE_list]
  #
  url = 'http://catalogo.bne.es/uhtbin/authoritybrowse.cgi?action=download&format=WBNMARA4&sel='
  url += '|'.join(BNE_list)
  print(url)
  response = requests.get(url=url, headers={'user-agent': user_agent})
  response.raise_for_status()
  content = response.text
  #
  genres = dict()
  for j,b in enumerate(BNE_list):
    genres[b]=dict()
    label = ''
    genre = ''
    ss = f'# {j+1}(.*?)\n(?:# {j+2}|</PRE>)'
    m = re.search(ss, content, re.DOTALL)
    if m is not None:
      # Genre:
      cc = m.group(1)
      m2 = re.search('^375\s+\$a([^$\n]+)', cc, re.MULTILINE)
      if m2 is not None:
        genre = m2.group(1)
      # Label (100 o 400)

      m2 = re.search('^100\s+.?\s+\$a([^$\n]+)', cc, re.MULTILINE)
      if m2 is not None:
        label = m2.group(1)
    genres[b]['label'] = label
    genres[b]['gender'] = genre
    #
  #return genres
  # Return a Pandas dataframe
  return pd.DataFrame.from_dict(genres, orient='index')

### Use the SUDOC Sparql endpoint to retrieve the gender of the records which
### identifiers are in SUDOC_list
def s_Gender(SUDOC_list, chunksize=1500, debug=False):
  """
  Use the SUDOC Sparql endpoint to retrieve the gender of the records which
  identifiers are in SUDOC_list.

  https://data.idref.fr/sparql

  :param SUDOC_list: A list with SUDOC identifiers
  :param chunksize:
   We think that the SPARQL Query API has a limit of 60 seconds to return any
   request. The larger the number of entities in the query, the higher the risk
   of reaching this limit, so it is necessary to make several requests with a
   reduced number of entities. The parameter chunksize sets the maximum number
   of entities to be sent in each query. If 500 requests error is returned, it
   is necessary to decreace the chunksize value.
  :return A Pandas data-frame
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  """
  if isinstance(SUDOC_list, str):
    SUDOC_list = [SUDOC_list]
  #
  n = len(SUDOC_list)
  # Number of entities exceeds chunksize:
  if n > chunksize:
    print(f"INFO: The number of entities ({n}) exceeds chunksize ({chunksize}).", sep="", file=sys.stderr)
    for k in range(int(n/chunksize)+1):
      offset = chunksize*k
      q_list = SUDOC_list[offset:offset+chunksize]
      if len(q_list) == 0:
        break
      print(f"\tINFO: Requesting entities from {offset+1} to {offset+len(q_list)}.", file=sys.stderr)
      d = s_Gender(q_list, chunksize=chunksize, debug=debug)
      if k==0:
        output = d
      else:
        output.update(d)
    return output
  #
  values = [f"<http://www.idref.fr/{x}/id>" for x in SUDOC_list]
  values = " ".join(values)

  url = 'https://data.idref.fr/sparql'
  #
  query = f"""SELECT DISTINCT ?sudoc ?label
(GROUP_CONCAT(DISTINCT ?sex;separator="|") as ?gender)
WHERE {{
  VALUES ?sudoc {{ {values} }}
  OPTIONAL {{?sudoc skos:prefLabel ?label.}}
  OPTIONAL {{?sudoc foaf:gender ?sex.}}
}} GROUP BY ?sudoc ?label
"""
  if debug:
    print(query, file=sys.stderr)
  #
  params = {'format': 'json',
            'query': query}
  response = requests.post(url=url, data=params, headers={'user-agent': user_agent})
  response.raise_for_status()
  j = response.json()
  bindings = j['results']['bindings']
  if len(bindings) == 0:
    return None
  #
  d = dict()
  for b in bindings:  # cada b es un dict
    sudoc = re.sub('^http://www.idref.fr/([^/]+)/id','\\1', b['sudoc']['value'])
    d[sudoc] = {}
    for k in ['label', 'gender']:
      d[sudoc][k] = ''
      if k in b:
        d[sudoc][k] = b[k]['value']
  # return d
  # Return a Pandas dataframe
  return pd.DataFrame.from_dict(d, orient='index')

### Use the GETTY Sparql endpoint to search for labels
def g_SearchLabel(label, debug=False):
  """
  Use the GETTY Sparql endpoint to search for labels
  https://vocab.getty.edu/queries#Finding_Subjects

  :param label: name of authors to search. It's better to use this format:
       last-name AND first-name (do not include year)
  :return A Pandas dataframe.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  """
  url = 'https://vocab.getty.edu/sparql'
  #
  query = f"""SELECT DISTINCT ?getty ?label ?gender
WHERE {{
  ?getty luc:term "{label}";
         skos:inScheme ulan:;
         gvp:parentStringAbbrev "Persons, Artists";
         gvp:prefLabelGVP/xl:literalForm ?label.
  OPTIONAL {{?getty foaf:focus/gvp:biographyPreferred/schema:gender/rdfs:label ?gender.
            FILTER(LANG(?gender)='en'). }}
}}
"""
  if debug:
    print(query, file=sys.stderr)
  #
  params = {'format': 'json',
            'query': query}
  response = requests.post(url=url, data=params, headers={'user-agent': user_agent})
  response.raise_for_status()
  j = response.json()
  bindings = j['results']['bindings']
  if len(bindings) == 0:
    return None
  #
  d = dict()
  for b in bindings:  # cada b es un dict
    sudoc = b['getty']['value'].replace('http://vocab.getty.edu/ulan/','')
    d[sudoc] = {}
    for k in ['label', 'gender']:
      d[sudoc][k] = ''
      if k in b:
        d[sudoc][k] = b[k]['value']
  # return d
  # Return a Pandas dataframe
  return pd.DataFrame.from_dict(d, orient='index')


### Use the GETTY Sparql endpoint to retrieve the gender of the records which
### identifiers are in GETTY_list
def g_Gender(GETTY_list, chunksize=10000, debug=False):
  """
  Use the GETTY Sparql endpoint to retrieve the gender of the records which
  identifiers are in GETTY_list.

  https://vocab.getty.edu/sparql

  :param GETTY_list: A list with GETTY identifiers
  :param chunksize:
   We think that the SPARQL Query API has a limit of 60 seconds to return any
   request. The larger the number of entities in the query, the higher the risk
   of reaching this limit, so it is necessary to make several requests with a
   reduced number of entities. The parameter chunksize sets the maximum number
   of entities to be sent in each query. If 500 requests error is returned, it
   is necessary to decreace the chunksize value.
  :return A Pandas data-frame
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  """
  if isinstance(GETTY_list, str):
    GETTY_list = [GETTY_list]
  #
  n = len(GETTY_list)
  # Number of entities exceeds chunksize:
  if n > chunksize:
    print(f"INFO: The number of entities ({n}) exceeds chunksize ({chunksize}).", sep="", file=sys.stderr)
    for k in range(int(n/chunksize)+1):
      offset = chunksize*k
      q_list = GETTY_list[offset:offset+chunksize]
      if len(q_list) == 0:
        break
      print(f"\tINFO: Requesting entities from {offset+1} to {offset+len(q_list)}.", file=sys.stderr)
      d = g_Gender(q_list, chunksize=chunksize, debug=debug)
      if k==0:
        output = d
      else:
        output.update(d)
    return output
  #
  values = "ulan:" + " ulan:".join(GETTY_list)

  url = 'https://vocab.getty.edu/sparql'
  #
  query = f"""SELECT DISTINCT ?getty ?label ?gender
WHERE {{
  VALUES ?getty {{ {values} }}
  OPTIONAL {{?getty gvp:prefLabelGVP/xl:literalForm ?label}}
  OPTIONAL {{?getty foaf:focus/gvp:biographyPreferred/schema:gender/rdfs:label ?gender.
            FILTER(LANG(?gender)='en').
           }}
}}
"""
  if debug:
    print(query, file=sys.stderr)
  #
  params = {'format': 'json',
            'query': query}
  response = requests.post(url=url, data=params, headers={'user-agent': user_agent})
  response.raise_for_status()
  j = response.json()
  bindings = j['results']['bindings']
  if len(bindings) == 0:
    return None
  #
  d = dict()
  for b in bindings:  # cada b es un dict
    sudoc = b['getty']['value'].replace('http://vocab.getty.edu/ulan/','')
    d[sudoc] = {}
    for k in ['label', 'gender']:
      d[sudoc][k] = ''
      if k in b:
        d[sudoc][k] = b[k]['value']
  return d


### Use a HHTP request to DNB catalog to retrive the gender of a record which
### identifier is known.
def d_Gender(DNB_id, debug=False):
  """
  Use the HTTP request to retrieve the gender of a records from DNV catalog.

  https://hub.culturegraph.org/entityfacts/DNB_id

  :param DNB_id: one idenfifier of DNB catalog
  :return A string with gender (in English).
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  """
  url = 'https://hub.culturegraph.org/entityfacts/' + DNB_id
  #
  if debug:
    print(url)
  #
  response = requests.get(url=url, headers={'user-agent': user_agent})
  response.raise_for_status()
  j = response.json()
  if 'gender' not in j:
    return ""
  m = re.search("([^#]+)$", j['gender']['@id'])
  return m.group(1)


#%% -- Simple text processing  ---------------------------------------------

def deaccenttext(text, excludechars='ñÑ'):
  """
  This function removes accents from the text (except for the letters in
  excludechars string) and returns a normalized Unicode string (NFKC).
  Note that the 'excludechars' string must be Unicode NFKC normalized.
  """
  result = ""
  text = unicodedata.normalize("NFKC", text)
  #
  for c in text:
    if c in excludechars:
      result += c
      continue
    #
    norm = unicodedata.normalize("NFKD", c)
    for ch in norm:
      if unicodedata.category(ch) not in ['Mn', 'Cc', 'Cf', 'Lm', 'So']:
        result += ch
  return unicodedata.normalize("NFKC", result)


def similar(a, b, deaccent=False, lower=False, order=False, mode="char", stops=set()):
  """
  Return SequenceMatcher.ratio() between strings a and b. If mode is "char"
  comparisons are made as char, else comparisons are made as terms (\w+). In
  mode "text" a set of stop words can be used in parameter 'stops'.
  """
  if deaccent:
    a = deaccenttext(a)
    b = deaccenttext(b)
  if lower:
    a = a.lower()
    b = b.lower()
  if mode != "char":
    a = re.findall("\w+", a)
    b = re.findall("\w+", b)
    if len(stops)>0:
      a = " ".join([x for x in a if x not in stops])
      b = " ".join([x for x in b if x not in stops])
  if order:
    if mode == 'char':
      a = " ".join(sorted(re.findall("\w+", a)))
      b = " ".join(sorted(re.findall("\w+", b)))
    else:
      a = sorted(a)
      b = sorted(b)
  #
  return SequenceMatcher(None, a, b).ratio()



#%% VIAF API ---------------------------------------------------------------
# VIAF API to search using HTTP.
# See https://www.oclc.org/developer/api/oclc-apis/viaf/authority-cluster.en.html
#
# Other APIs: https://www.oclc.org/developer/api/oclc-apis.en.html

def v_Autosuggest(author, deaccent=True):
  """
  Search the name of the author from the VIAF AutoSuggest API and returns
  information in JSON format of the records found. Note that only returns a
  maximum of 10 records.

  See: https://www.oclc.org/developer/api/oclc-apis/viaf/authority-cluster.en.html

  Note that records returned by Autosuggest API are not VIAF cluster records.
  Each returned record in 'result' claim contains the sources (authority
  catalog or database) and the identifier in them. In contrast, a VIAF record
  is considered a "cluster record," which is the result of combining records
  from many sources: libraries around the world into a single record.

  Author is deaccentered before search if deaccent=True.

  :param author: String to search for author. Please, see the structure of the
         author string to obtain better results:
         author:  last name, first name[,] [([year_of_bird][-year_of_death])]

  :param deaccent: String is deaccentered before search if deaccent=True.
  :return A list with the 'result' claims or None.
          Note that the same author can be more than one time if he/she has
          more than one name in VIAF.
  :raise Exception: From response.raise_for_status() or other exception.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :example
  >>> results = v_Autosuggest('Albaladejo, Luis')
  >>> # It is easy conveting as Pandas dataframe
  ... df = pd.DataFrame.from_dict(results)
  """
  if deaccent:
    author = deaccenttext(author)
  #
  # headers = {'user-agent': user_agent}  # Not necessary for VIAF API
  url = "http://www.viaf.org/viaf/AutoSuggest"
  params = {'query': author}
  response = requests.get(url=url, params=params) #, headers=headers)
  response.raise_for_status()
  j = response.json()
  if 'result' in j:
    return j['result']
  return None

def v_AutosuggestPersonal(author, deaccent=True):
  """
  Return only the results of type "Personal" using the VIAF AutoSuggest API.

  :param author: String to search for author. Please, see the structure of the
         author string to obtain better results:
         author:  last name, first name[,] [([year_of_bird][-year_of_death])]
  :param deaccent: String is deaccentered before search if deaccent=True.
  :return A list with the 'result' claims of nametype='personel' or None.
          Note that the same author can be more than one time if he/she has
          more than one name in VIAF.
  :raise Exception: From response.raise_for_status() or other exception.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :example
  >>> results = v_AutosuggestPersonal('Albaladejo, Luis')
  >>> # It is easy conveting as Pandas dataframe
  ... df = pd.DataFrame.from_dict(results)
  """
  response = v_Autosuggest(author, deaccent)
  if response is None:
    return None
  output = list()
  for record in response:
    if record['nametype'] == 'personal':
      output.append(record)
  return output

### Search interface
def v_Search(CQL_query, schema='JSON', start=1, nmax=30, debug=False):
  """
  Run the CQL_query using the VIAF Search API and returns a list of records
  found. The search string is formed using the CQL_query syntax of the API.
  Note that returned records use the record schema set in the recordSchema
  parameter. Record schema values that are valid for our proccesing are:
  - 'info:srw/schema/1/JSON'  [very similar to http://viaf.org/VIAFCluster]
  - 'http://viaf.org/BriefVIAFCluster'

  The brief schema not includes birthDate, deathDate, gender and occupation,
  but is fast to check if titles (works) match.

  If the number of records found is greater than 250 (API restrictions),
  successive requests are made.
  See https://www.oclc.org/developer/api/oclc-apis/viaf/authority-cluster.en.html

  The relational operator in search must be included in the CQL_query. The API
  supports:

    =     (one or more terms in the same field (title/name/alternative))
    exact (search the exact string, including any punctuation)
    any   (for any of a list of terms)
    all   (for all the listed terms in any field (title/name/alternative))
    <
    >
    <=
    >=
    not

  For example:
    'local.names = "Diaz Francisco', searching in each field 1xx, 4xx, 5xx.
    'local.names all "Diaz Francisco', searching is in concatenation of all
                                      the fields 1xx, 4xx, 5xx (word 'Diaz' and
                                      'Francisco' can be in differents fields)

  :param CQL_query: String with the search in CQL language.
  :param recordSchema: The schema of the record. Only supports 'JSON' or 'brief'.
         If schema=JSON, then recordSchema='info:srw/schema/1/JSON' (default).
         If schema=brief, then recordSchema='http://viaf.org/BriefVIAFCluster'
  :param nmax: Maximun number of record returned.
  :return A dict with the records found {viafId : record, viafID: record}
  :raise Exception: From response.raise_for_status() or other exception.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  """
  maxrecords = nmax if nmax < VIAF_LIMIT else VIAF_LIMIT

  url = "https://www.viaf.org/viaf/search"
  # headers = {'user-agent': user_agent}  # Not necessary for VIAF API

  if schema == 'JSON':
    recordSchema = 'info:srw/schema/1/JSON'
  elif schema == 'brief':
    recordSchema = 'http://viaf.org/BriefVIAFCluster'
  else:
    print("ERROR in v_Search(): recordSchema parameter is not valid.", file=sys.stderr)
    return None

  # Record schema values that are valid for our proccesing are:
  # 'info:srw/schema/1/JSON'  [very similar to http://viaf.org/VIAFCluster]
  # and 'http://viaf.org/BriefVIAFCluster', but last not includes birthDate,
  # deathDate, gender and occupation.
  params = {'httpAccept' : 'application/json',
            'maximumRecords' : maxrecords,
            'recordSchema'   : recordSchema,
            # 'recordSchema' : 'info:srw/schema/1/JSON',
            # 'recordSchema' : 'http://viaf.org/VIAFCluster',
            # 'recordSchema' : 'http://viaf.org/BriefVIAFCluster',
            # 'recordSchema' : 'info:srw/schema/1/JSONLINKS',
            # 'recordSchema' : 'info:srw/schema/1/unimarc-v0.1',
            # 'recordSchema' : 'http://www.w3.org/1999/02/22-rdf-syntax-ns',
            'startRecord' : start,
            'query' : CQL_query}
  #
  try:
    output = dict()
    while True:
      response = requests.get(url=url, params=params) #, headers=headers)
      if debug:
        print(requests.utils.unquote(response.url), file=sys.stderr)
      response.raise_for_status()
      j = response.json()
      nrecords = int(j['searchRetrieveResponse']['numberOfRecords'])
      if nrecords == 0:
        return output
      #
      nrecords = nrecords + 1 - start
      records = j['searchRetrieveResponse']['records']
      records = [x['record']['recordData'] for x in records]
      if schema == 'JSON':
        output.update({x['viafID']:x for x in records})
      else:
        output.update({x['viafID']['#text']:x for x in records})
      #
      if len(output) >= nmax or len(output) >= int(nrecords): # Break the while -> return
        if debug and params['startRecord'] > start:
          print(f' INFO: Retrieved {len(output)} records.', file=sys.stderr)
        return output
      #
      if debug:
        if params['startRecord'] == start:  # first time, inf
          print(f"INFO: Number of records found ({nrecords}) excedes the maximun per request API limit ({maxrecords}). Doing successivelly requests.", file = sys.stderr)
        print(f" INFO: Retrieved {len(output)} records.", file=sys.stderr)
      # New requests: start in maxrecords
      params['startRecord'] += maxrecords
      # Decrease nmax and retrive maxrecords (nmax if nmax>VIAF_LIMIT)
      nmax = nmax - maxrecords
      maxrecords = nmax if nmax < VIAF_LIMIT else VIAF_LIMIT
      params['maximumRecords'] = maxrecords
  #
  except Exception as ex:
    print(f'Error in "v_Search": {ex}', file=sys.stderr)
    print(f'Error in "v_Search": response.status_code = {response.status_code}', file=sys.stderr)
    print(f'Error in "v_Search": {CQL_query}', file=sys.stderr)
    if response.status_code == 429:
      t = response.headers['Retry-after']
      print(f'Error 429: \'Retry-after: {t}\'', file=sys.stderr)
      m = re.match(r'^\d+$', t)
      if m is None:
        t = int(http2time(t) - int(time()))
      else:
        t = int(t)
      if t > 600:
        raise("ERROR: receive a 429 status-code response, but retry-after > 600")
      print(f"Received a 429 status-code response. Sleeping {t} seconds",
            file=sys.stderr)
      sleep(t)
    return None
    # return []

### Search a string in any Field
def v_SearchAnyField(string, op="=", schema='JSON', start=1, nmax=30, debug=False):
  """
  Search 'string' in all fields, case insensitive, using the operator "op"
  (defaults "=").

  This function is a wrapper to v_Search, using this CQL_Query:
    'cql.any op "string"'

  :param string: It is the string to search.
  :param op: The operator used in the search.
  :param schema: The schema of the record. Only supports 'JSON' or 'brief'.
  :param nmax: Maximun number of record returned.
  :return A dict with the records found.
  :raise Exception: From response.raise_for_status() or other exception.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  """
  string = string.replace('"', "'")
  CQL_query = 'cql.any ' + op + ' "' + string + '"'
  return v_Search(CQL_query, schema=schema, start=start, nmax=nmax, debug=debug)

### Search for author names
def v_SearchByName(name, mode='personalNames', op="=", schema='JSON', start=1,
                  nmax=30, debug=False):
  """
  Search for names of author.
  This function is a wrapper to v_Search, using one of this modes:

    'local.mainHeadingEl op "name"'
    'local.names op "name"'
    'local.personalNames op "name"'

  :param name: It is the name to search for. For betters results use one of
         this format for the name:

    - If year of bird or death is not known:
        name: last-name, first-name
        name: first-name last-name

    - If year of bird or death is not known:
      last-name, first-name, year_of_bird-year_of_death

  :param mode: mode applied in the search:

    - 'mainHeadingEl': Search preferred Name - names which are the preferred
       form in an authority record (1xx fields of the MARC records):
        1xx: Main Entry: Personal name (100), corporate name (110), meeting
             name (111), uniform title (130)

    - 'names': Search Names within the authority record (1xx, 4xx, 5xx fields)
        1xx: Heading: Personal name (100), corporate name (110), meeting
                      name (111), uniform title (130), Events, ...
        4xx: See From Tracing: Personal name (400), corporate name (410),
             meeting name (411), uniform title (430), Events...
        5xx: See Also From Tracing: Personal name (500), corporate name (510),
             meeting name (511), uniform title (530), Events...

    - 'personalNames': Search Personal Names within the authority record:
        100: Heading - Personal Name (NR)
        400: See From Tracing - Personal Name (R)
        500: See Also From Tracing - Personal Name (R)

  :param op: The operator used in the search.
  :param schema: The schema of the record. Only supports 'JSON' or 'brief'.
  :param nmax: Maximum number of VIAF records to return.
  :return A list with the records found.
  :raise Exception: From response.raise_for_status() or other exception.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :example
  >>> j = v_SearchByName('Albaladejo, Luis', mode='personalNames')
  ... viaf = j[list(j.keys())[0]]   # Get first record
  ... g = v_sources(viaf)
  """
  name = name.replace('"', "'")
  if mode== 'mainHeadingEl':
    CQL_query = 'local.mainHeadingEl ' + op + ' "' + name + '"'
  elif mode == 'names':
    CQL_query = 'local.names ' + op + ' "' + name + '"'
  elif mode == 'personalNames':
    CQL_query = 'local.personalNames ' + op + ' "' + name + '"'
  else:
    raise ValueError(f"ERROR in v_Search(): mode '{mode}' is invalid.")
  #
  return v_Search(CQL_query, schema=schema, start=start, nmax=nmax, debug=debug)

### Search for titles in VIAF
def v_SearchByTitle(title, op="=", schema='JSON', start=1, nmax=30, debug=False):
  """
  This function is a wrapper to v_Search, using this CQL_Query:
    'local.title all "title"'
  The search is performed with the "=" operator by default, case insentive.

  :param title: It is the title to search.
  :param op: The operator used in the search.
  :param schema: The schema of the record. Only supports 'JSON' or 'brief'.
  :param nmax: Maximum number of VIAF records to return.
  :return A list with the records found.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :example
  >>> j = v_SearchByTitle('La vida perdularia', op='exact')
  ... viaf = j[list(j.keys())[0]]   # Get first record
  ... titles = v_titles(viaf)
  """
  title = title.replace('"', "'")
  CQL_query = 'local.title ' + op + ' "' + title + '"'
  return v_Search(CQL_query, schema=schema, start=start, nmax=nmax, debug=debug)

### Get a record from VIAF
def v_GetRecord(viafid, record_format='viaf.json', check=False):
  """
  Obtain the record cluster identified by 'viafid' from VIAF, in the format
  indicated in 'record_format'. Note that the returned record may be a VIAF
  cluster record or a redirect/scavenged record: the function returns a tuple
  with original, redirect or scavenged record.

  :param viafid The VIAF identifier.
  :param record_format 'viaf.json' (default) or "viaf.xml" (please, see
         https://www.oclc.org/developer/api/oclc-apis/viaf/authority-cluster.en.html)
  :param check: If check=True and the record is a redirect/scavenged, then
         returns a tuple with the ('redirection', viafid_of_the_redirection) or
         ('scavenged', record_scavenged). If record is not a redirect or
         scavenged record, then returns a tuple with ('original', record).
         If check=False (default), only return the VIAF record, if any.
  :return A VIAF record or a tuple.
  :raise Exception: From response.raise_for_status() or other exception.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :example
  >>> viaf = v_GetRecord('29550309')
  """
  #
  viafid = f"{viafid}"  # Perhaps viafid is an int
  # headers = {'user-agent': user_agent} # Not necessary for VIAF API
  url = "http://viaf.org/viaf/" + viafid + '/' + record_format
  #
  # response = requests.get(url=url, params=params, headers=headers)
  response = requests.get(url=url) # headers=headers)
  response.raise_for_status()
  if record_format == 'viaf.xml':
    text = response.text
    # Detect it is a redirection
    if check is True:
      m = re.search(r'<ns0:directto>([^<]+)</ns0:directto>', text)
      if m is not None:
        viafid = m.group(1)
        _, text = v_GetRecord(viafid, record_format, check)
        return ('redirect', text)
      m = re.search(r'<ns0:scavenged>([^<]+)</ns0:scavenged>', text)
      if m is not None:
        return ('scavenged', m.group(1))
      return ('original', text)
    return(text)
  #
  j = response.json()
  if check is True:
    if 'redirect' in j:
      viafid = j['redirect']['directto']
      _, j = v_GetRecord(viafid, record_format, check)
      return ('redirect', j)
    if 'scavenged' in j:
      return ('scavenged', j['scavenged']['VIAFCluster'])
    return ('original', j)
  return(j)

### Return a MARC21 or Unimarc record  record from source
def v_GetProcessed(record_id, source='lc'):
  """
  Return a MARC21 or Unimarc record from the authority source using the
  identifier in that source.

  See: https://www.oclc.org/developer/api/oclc-apis/viaf/authority-source.en.html

  :param record_id: The identifier in the authority source.
  :param source: Authority source (library).
  :raise Exception: From response.raise_for_status() or other exception.
  :return The record in MARC21 or Unimarc Name Authority records format.
  :raise Exception: From response.raise_for_status() or other exception.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :example
  >>> record = v_GetProcessed('XX5637072', 'BNE')
  """
  if record_id == '':
    return None
  #
  if isinstance(record_id, int):
    record_id = str(record_id)
  #
  url = "http://viaf.org/processed/" + source + '|' + record_id
  headers = {#'user-agent': user_agent,  # Not necessary for VIAF API
             'accept' : 'application/marc21+xml'}
  response = requests.get(url=url, headers=headers)
  try:
    response.raise_for_status()
  except Exception as ex:
    print("Excepción en v_Processed", ex)
    return ''
  return response.text

### Check if a VIAF record is "Personal", i.e., about a person, not a book,
### not an organization, etc.
def v_isPersonal(viaf):
  """
  Return True if the VIAF record has nameType == "Personal"
  :param viaf: A VIAF cluster record (in JSON format)
  :raise Exception: From response.raise_for_status() or other exception.
  :return True if the VIAF record has nameType == "Personal", else False
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :example
  >>> j = v_SearchAnyField('Pérez García, Luis', namx=5)
  ... for vid, viaf in j.items():
  ...     print(f"{vid} \t isPersonal:{v_isPersonal(viaf)}")
  """
  if 'nameType' in viaf:
    if viaf['nameType'] == 'Personal':
      return True
    if '#text' in viaf['nameType'] and viaf['nameType']['#text'] == 'Personal':
      return True
  return False

def v_titles(viaf, normNFKC=True):
  """
  Return titles of works from the VIAF record. Note that the VIAF record musts
  be in JSON format.

  :param viaf: A VIAF cluster record (in JSON format)
  :param normNFKC: If Unicode NFKC normalization must be applied on titles.
  :return A list of titles of works.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :example
  >>> j = v_SearchByName('Pérez García, Luis', mode='personalNames')
  ... viaf = j[list(j.keys())[0]]   # Get first record
  ... titles = v_titles(viaf)
  """
  titles = list()
  if ('titles' in viaf and viaf['titles'] is not None
      and 'work' in viaf['titles'] and viaf['titles']['work'] is not None):
    # viaf['titles']['work'] puede ser 1 dict o una lista de dict
    if isinstance(viaf['titles']['work'], dict):
      tit = viaf['titles']['work']['title']
      titles.append(str(tit))
    else:
      for t in viaf['titles']['work']:
        tit = t['title']
        if isinstance(tit, list):
          for t2 in tit:
            titles.append(str(t2))
        else:
          titles.append(str(tit))
    if normNFKC:
      titles = [unicodedata.normalize("NFKC", x) for x in titles]
  return titles

def v_gender(viaf):
  """
  Return the gender of the author from the VIAF record. Note that the VIAF
  record musts be in JSON format.

  :param viaf: A VIAF cluster record (in JSON format)
  :return The gender: female, male, u (undefined) or "" (no gender field in record.)
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :example
  >>> j = v_SearchByName('Pérez García, Luis', mode='personalNames')
  ... for vid, viaf in j.items():
  ...     print(f"{vid} \t gender:{v_gender(viaf)}")
  """
  if 'fixed' in viaf:
    if viaf['fixed']['gender'] == 'a':
      return "female"
    if viaf['fixed']['gender'] == 'b':
      return "male"
    return viaf['fixed']['gender']
  return ""

def v_dates(viaf):
  """
  Return bird year and death year from the VIAF cluster record with this
  format byear:dyear. Note that the VIAF record musts be in JSON format.

  :param viaf: VIAF cluster record (in JSON format).
  :return The bird year and death year in the form: byear:dyear.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :example
  >>> j = v_SearchByName('Pérez García, Luis', mode='personalNames')
  ... for vid, viaf in j.items():
  ...     print(f"{vid} \t dates={v_dates(viaf)}")
  """
  byear = dyear = ''
  if 'birthDate' in viaf:
    byear = viaf['birthDate'][:4]
    if byear == '0':
      byear = ''
  if 'deathDate' in viaf:
    dyear = viaf['deathDate'][:4]
    if dyear == '0':
      dyear = ''
  return byear + ':' + dyear

def v_occupations(viaf, normNFKC=True):
  """
  Return the occupations from the VIAF record. Note that the VIAF record musts
  be in JSON format.

  :param viaf: VIAF cluster record (in JSON format).
  :param normNFKC: If Unicode NFKC normalization must be applied on occupations.
  return: A list with the occupations.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :example
  >>> j = v_SearchByName('Pérez García, Luis', mode='personalNames')
  ... viaf = j['48071603']
  ... occupation = v_occupations(viaf)
  """
  occs = []
  sources = ['JPG', 'LC', 'BNE']
  if 'occupation' in viaf and 'data' in viaf['occupation']:
    # viaf['occupation']['data'] puede ser 1 dict o una lista de dict
    if isinstance(viaf['occupation']['data'], dict):
      oc = viaf['occupation']['data']['text']
      so = viaf['occupation']['data']['sources']
      if so['s'] in sources:
        occs.append(oc)
    else:
      for t in viaf['occupation']['data']:
        oc = t['text']
        so = t['sources']
        if isinstance(so['s'], list):
          for s in so['s']:
            if s in sources:
              occs.append(str(oc))
              break # para que no tome oc varias veces si están 'JPG' y 'LC'
        else:
          if so['s'] in sources:
            occs.append(str(oc))
    if normNFKC:
      occs = [unicodedata.normalize("NFKC",x) for x in occs]
  return occs

def v_sources(viaf, normNFKC=True):
  """
  Return the text of all sources id from the VIAF record using the
  mainHeadings data. Note that the VIAF record musts be in JSON format.

  :param viaf: VIAF cluster record (in JSON format).
  :param normNFKC: If Unicode NFKC normalization must be applied on author names.
  :return A dict with all text (name) and sources.
       text1 -> dict: {source1: id , source2: id, ...}
       text2 -> dict: {source1: id , source2: id, ...}
       ...
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :example
  >>> j = v_SearchByName('Pérez García, Luis', mode='personalNames')
  ... viaf = j[list(j.keys())[0]]   # Get first record
  ... sources = v_sources(viaf)
  >>> # Convert the dict returned to a Pandas data-frame:
  ... df = pd.DataFrame.from_dict(sources, orient='index')
  """
  texts = dict()
  vv = viaf['mainHeadings']['data']
  if not isinstance(vv, list): # Solo un elemento, lo pongo como list
    vv = [vv]
  for l in vv:                # cada l es un dict()
    sources = l['sources']['sid']
    if not isinstance(sources, list):
      sources = [sources]
    name = l['text']
    s = dict()
    # Add viafID itself
    # s['viafID'] = viaf['viafID']
    for source in sources:
      library, ident = source.split('|')
      s[library] = ident
    texts[name] = s
  #
  if normNFKC:
    texts = {unicodedata.normalize("NFKC",x):y for x,y in texts.items()}
  return texts

def v_sourceId(viaf, source, normNFKC=True):
  """
  Return the text and the identifier that the VIAF record has in the source(s).
  Note that the VIAF record musts be in JSON format.

  :param viaf: The VIAF cluster record (in JSON format).
  :param source: The source (LC, WKP, JPG, BNE...). More than one source can
         be set, using '|' as separator.
  :param normNFKC: If Unicode NFKC normalization must be applied on author names.
  :return A dict with keys the source(s) and values a tuple (term, identifier)
          if source is in the VIAF cluster record. If the source does no exists
          in the viaf record, return a void tuple for the source.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :example
  >>> j = v_SearchByName('Pérez García, Luis', mode='personalNames')
  ... viaf = j[list(j.keys())[0]]   # Get first record
  ... sources = v_sourceId(viaf, 'BNE|LC')
  >>> # Convert the dict returned to a Pandas data-frame:
  ... df = pd.DataFrame.from_dict(sources, orient='index')
  """
  texts = v_sources(viaf, normNFKC=normNFKC)
  sources = {x:None for x in source.split('|')}
  for text,idents in texts.items():
    for library, ident in idents.items():
      for s in source.split('|'):
       if library == s:
         sources[s] = (text, ident)
  return sources

def v_sourcesX400(viaf, x='x400', normNFKC=True):
  """
  Return the normalized text of all sources id from the VIAF record using the
  x400s/x500s data of the VIAF cluster record.
  Note that the VIAF record musts be in JSON format.

  :param viaf: VIAF cluster record (in JSON format).
  :param x: If must be used x400 or x500 source.
  :param normNFKC: If Unicode NFKC normalization must be applied on author names.
  :return A dict with all text (name) and sources:
         text1 -> dict: {source1: id , source2: id, ...}
         text2 -> dict: {source1: id , source2: id, ...}
         ...
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :example:
  >>> j = v_SearchByName('Pérez García, Luis', mode='personalNames')
  ... viaf = j[list(j.keys())[0]]    # Get first record
  ... sources = v_sourcesX400(viaf)
  >>> # Convert the dict returned to a Pandas data-frame:
  ... df = pd.DataFrame.from_dict(sources, orient='index')
  """
  texts = dict()
  xx = x + 's'
  if xx not in viaf:
    return texts
  vv = viaf[xx][x]
  if not isinstance(vv, list): # Solo un elemento, lo pongo como list
    vv = [vv]
  for v in vv:                # cada l es un dict()
    text = v['datafield']['normalized']
    sources = v['sources']['sid']
    if not isinstance(sources, list): # Solo un elemento, lo pongo como list
      sources = [sources]
    s = dict()
    for source in sources:
      library, ident = source.split('|')
      s[library] = ident
    texts[text] = s
  #
  if normNFKC:
    texts = {unicodedata.normalize("NFKC",x):y for x,y in texts.items()}
  return texts

def v_coauthors(viaf, normNFKC=True):
  """
  Return the coauthors from the VIAF record. Note that the VIAF record musts
  be in JSON format.

  :param viaf: VIAF cluster record (in JSON format).
  :param normNFKC: If Unicode NFKC normalization must be applied on author names.
  :return A dict inwith coauthors as keys and the count of works they
          colaborate as values.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :example:
  >>> j = v_SearchByName('Pérez García, Luis', mode='personalNames')
  ... viaf = j[list(j.keys())[0]]    # Get first record
  ... coauthors = v_coauthors(viaf)
  """
  coauthors = dict()
  if 'coauthors' in viaf and 'data' in viaf['coauthors']:
    # viaf['coauthors']['data'] puede ser 1 dict o una lista de dict
    if isinstance(viaf['coauthors']['data'], dict):
      author = viaf['coauthors']['data']['text']
      count  = viaf['coauthors']['data']['@count']
      coauthors[author] = count
    else:
      for t in viaf['coauthors']['data']:
        author = t['text']
        count  = t['@count']
        coauthors[author] = count
    #
    if normNFKC:
      coauthors = {unicodedata.normalize("NFKC",x):y for x,y in coauthors.items()}
  return coauthors

def v_wikipedias(viaf):
  """
  Return the Wikipedia pages (URL) from the VIAF record. Note that the VIAF
  record musts be in JSON format.

  :param viaf: The VIAF cluster record (in 1/JSON or BriefVIAFCluster format).
  :return A list with the URL of the Wikipedia pages in the VIAF cluster
          record or a void list if no Wikipedia pages are in record.
  :raise Exception: From response.raise_for_status() or other exception.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  :example:
  >>> j = v_SearchByName('Pérez García, Luis', mode='personalNames')
  ... viaf = j[list(j.keys())[0]]    # Get first record
  ... wkp = v_wikipedias(viaf)
  """
  if 'xLinks' not in viaf or viaf['xLinks'] is None or 'xLink' not in viaf['xLinks']:
    return []
  #
  wikis = list()
  xLinks = viaf['xLinks']['xLink']
  if '#text' in xLinks:  # Only one => set as list
    xLinks = [xLinks]
  for l in xLinks:
    url = l['#text']
    m = re.match('https?://[^.]+.wikipedia.org', url)
    if m is not None:
      wikis.append(url)
  return wikis

def v_allinfo(viaf, normNFKC=True):
  """
  Returns all data of interest from the VIAF record. Note that the VIAF
  record musts be in JSON format.

  :param viaf: The VIAF cluster record (in 1/JSON or BriefVIAFCluster format).
  :param normNFKC: If Unicode NFKC normalization must be applied on names,
         titles and occupations.
  :return A dict.
  :author Angel Zazo, Department of Computer Science and Automatics, University of Salamanca
  """
  return {
    'viafId': viaf['viafID'],
    'gender': v_gender(viaf),
    'dates' : v_dates(viaf),
    'sources':v_sources(viaf, normNFKC=normNFKC),
    'sourcesX400':v_sourcesX400(viaf, normNFKC=normNFKC),
    'titles': v_titles(viaf, normNFKC=normNFKC),
    'occupations' : v_occupations(viaf, normNFKC=normNFKC),
    'coauthors' : v_coauthors(viaf, normNFKC=normNFKC),
    'wikipedias': v_wikipedias(viaf)
    }

