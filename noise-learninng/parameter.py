input_path = "/DirectoryName/Appendix/input/OOsmells/"
pred_path = "/DirectoryName/noise-learninng/"
ant = ["ant/rel/1.6.0", "ant/rel/1.6.1", "ant/rel/1.6.2", "ant/rel/1.6.3", "ant/rel/1.6.4", "ant/rel/1.7.0",
       "ant/rel/1.7.1", "ant/rel/1.8.1", "ant/rel/1.8.2", "ant/rel/1.8.3"]
argouml = ["argouml/VERSION_0_12", "argouml/VERSION_0_14", "argouml/VERSION_0_18_1", "argouml/VERSION_0_20",
           "argouml/VERSION_0_22", "argouml/VERSION_0_24", "argouml/VERSION_0_26", "argouml/VERSION_0_30",
           "argouml/VERSION_0_30_1", "argouml/VERSION_0_30_2", "argouml/VERSION_0_32_1", "argouml/VERSION_0_32_2",
           "argouml/VERSION_0_34"]
cassandra = ["cassandra/cassandra-0.7.0", "cassandra/cassandra-0.7.2", "cassandra/cassandra-0.7.3",
             "cassandra/cassandra-0.8.0", "cassandra/cassandra-0.8.1", "cassandra/cassandra-0.8.3",
             "cassandra/cassandra-1.0.0", "cassandra/cassandra-1.1.0"]
derby = ["derby/10.1.3.1", "derby/10.2.2.0", "derby/10.3.3.0", "derby/10.4.2.0", "derby/10.5.3.0", "derby/10.6.2.1",
         "derby/10.7.1.1", "derby/10.8.2.2", "derby/10.9.1.0"]
eclipse = ["eclipse/R2_0", "eclipse/R2_1", "eclipse/R2_1_1", "eclipse/R2_1_2", "eclipse/R2_1_3", "eclipse/R3_0",
           "eclipse/R3_0_1", "eclipse/R3_0_2", "eclipse/R3_1", "eclipse/R3_1_1", "eclipse/R3_1_2", "eclipse/R3_2",
           "eclipse/R3_2_1", "eclipse/R3_2_2", "eclipse/R3_3", "eclipse/R3_3_1", "eclipse/R3_3_1_1", "eclipse/R3_3_2",
           "eclipse/R3_4", "eclipse/R3_4_1", "eclipse/R3_4_2"]
elasticsearch = ["elasticsearch/v0.12.0", "elasticsearch/v0.13.0", "elasticsearch/v0.14.0", "elasticsearch/v0.15.0",
                 "elasticsearch/v0.16.0", "elasticsearch/v0.17.0", "elasticsearch/v0.18.0", "elasticsearch/v0.19.0"]
hadoop = ["hadoop/release-0.1.0", "hadoop/release-0.2.0", "hadoop/release-0.3.0", "hadoop/release-0.4.0",
          "hadoop/release-0.5.0", "hadoop/release-0.6.0", "hadoop/release-0.7.0", "hadoop/release-0.8.0",
          "hadoop/release-0.9.0"]
hsqldb = ["hsqldb/2.0.0", "hsqldb/2.2.0", "hsqldb/2.2.1", "hsqldb/2.2.2", "hsqldb/2.2.3", "hsqldb/2.2.4",
          "hsqldb/2.2.5", "hsqldb/2.2.6", "hsqldb/2.2.7", "hsqldb/2.2.8"]
incubating = ["incubating/release-0.1-incubating", "incubating/release-0.2-incubating",
              "incubating/release-0.3-incubating", "incubating/release-0.4-incubating",
              "incubating/release-0.5-incubating", "incubating/release-0.6"]
nutch = ["nutch/release-0.7", "nutch/release-0.8", "nutch/release-0.9", "nutch/release-1.1", "nutch/release-1.2",
         "nutch/release-1.3", "nutch/release-1.4"]
qpid = ["qpid/0.10", "qpid/0.12", "qpid/0.14", "qpid/0.16", "qpid/0.18"]
wicket = ["wicket/wicket-1.4.11", "wicket/wicket-1.4.13", "wicket/wicket-1.4.14", "wicket/wicket-1.4.15",
          "wicket/wicket-1.4.16", "wicket/wicket-1.4.17", "wicket/wicket-1.4.18", "wicket/wicket-1.4.19",
          "wicket/release/wicket-1.4.20"]
xerces = ["xerces/Xerces-J_1_0_4", "xerces/Xerces-J_1_2_0", "xerces/Xerces-J_1_2_1", "xerces/Xerces-J_1_2_2",
          "xerces/Xerces-J_1_2_3", "xerces/Xerces-J_1_3_0", "xerces/Xerces-J_1_3_1", "xerces/Xerces-J_1_4_0",
          "xerces/Xerces-J_1_4_1", "xerces/Xerces-J_1_4_2"]
projectNames = [argouml, ant, # cassandra,
                derby, eclipse, elasticsearch, hadoop, hsqldb, incubating, nutch,
                qpid, wicket, xerces]
smellNames = ["GodClass", "ComplexClass", "ClassDataShouldBePrivate", "SpaghettiCode", "LongMethod",
              # "FeatureEnvy",
              "InappropriateIntimacy",
              # "MiddleMan",
              # "RefusedBequest",
              #"SpeculativeGenerality",
              "LongParameterList"
              ]


def return_path(smell_name, project):
    path = "{0}{1}/{2}.csv".format(input_path, smell_name, project.replace('/', '-'))
    return path


def return_paths(smell_name, projects):
    path_list = []
    for project in projects:
        path = return_path(smell_name, project)
        path_list.append(path)
    return path_list
