{
 "0": {
  "timestep": 0,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 0,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "eng_station",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "eng_station",
    "application",
    "web-browser",
    "execute"
   ],
   "response": {
    "status": "failure",
    "data": {}
   },
   "reward": 0.01,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 0,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 0,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 0,
   "action": "node-file-scan",
   "parameters": {
    "node_name": "robot_controller",
    "folder_name": "program",
    "file_name": "robot_program.bin"
   },
   "request": [
    "network",
    "node",
    "robot_controller",
    "file_system",
    "folder",
    "program",
    "file",
    "robot_program.bin",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.8510000000000001,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 1
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 0,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 0,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "1": {
  "timestep": 1,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 1,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 1,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 1,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "office_pc",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "office_pc",
    "application",
    "web-browser",
    "execute"
   ],
   "response": {
    "status": "failure",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 1,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.8,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 1
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 0
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 1,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 1,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "2": {
  "timestep": 2,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 2,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 2,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 2,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 2,
   "action": "node-network-service-recon",
   "parameters": {
    "source_node": "eng_station",
    "target_ip_address": [
     "10.2.20.0/24"
    ],
    "target_port": 22,
    "target_protocol": "tcp",
    "show": true
   },
   "request": [
    "network",
    "node",
    "eng_station",
    "application",
    "nmap",
    "network_service_recon",
    {
     "target_ip_address": [
      "10.2.20.0/24"
     ],
     "target_port": 22,
     "target_protocol": "tcp",
     "show": true
    }
   ],
   "response": {
    "status": "success",
    "data": {
     "10.2.20.40": {
      "tcp": [
       22
      ]
     },
     "10.2.20.10": {
      "tcp": [
       22
      ]
     },
     "10.2.20.20": {
      "tcp": [
       22
      ]
     },
     "10.2.20.30": {
      "tcp": [
       22
      ]
     },
     "10.2.20.1": {
      "tcp": [
       22
      ]
     }
    }
   },
   "reward": 0.8500000000000001,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 0
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 2,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 2,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "3": {
  "timestep": 3,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 3,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 3,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 3,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 3,
   "action": "node-file-scan",
   "parameters": {
    "node_name": "robot_controller",
    "folder_name": "program",
    "file_name": "robot_program.bin"
   },
   "request": [
    "network",
    "node",
    "robot_controller",
    "file_system",
    "folder",
    "program",
    "file",
    "robot_program.bin",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.8500000000000001,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 1,
          "outbound": 1
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 1,
           "outbound": 1
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 1,
          "outbound": 1
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 1,
           "outbound": 1
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 1,
          "outbound": 1
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 1,
           "outbound": 1
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 1,
          "outbound": 1
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 1,
           "outbound": 1
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 1,
          "outbound": 1
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 1,
           "outbound": 1
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 1
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 3,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 3,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "4": {
  "timestep": 4,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 4,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 4,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 4,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 4,
   "action": "router-acl-remove-rule",
   "parameters": {
    "target_router": "edge_router",
    "position": 5
   },
   "request": [
    "network",
    "node",
    "edge_router",
    "acl",
    "remove_rule",
    5
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.8500000000000001,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 0
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 4,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 4,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "5": {
  "timestep": 5,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 5,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "eng_station",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "eng_station",
    "application",
    "web-browser",
    "execute"
   ],
   "response": {
    "status": "failure",
    "data": {}
   },
   "reward": 0.01,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 5,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "robot_historian",
    "application_name": "database-client"
   },
   "request": [
    "network",
    "node",
    "robot_historian",
    "application",
    "database-client",
    "execute"
   ],
   "response": {
    "status": "failure",
    "data": {}
   },
   "reward": -0.5,
   "reward_info": {
    "connection_attempt_status": "failure"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 5,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 5,
   "action": "node-file-scan",
   "parameters": {
    "node_name": "robot_controller",
    "folder_name": "program",
    "file_name": "robot_program.bin"
   },
   "request": [
    "network",
    "node",
    "robot_controller",
    "file_system",
    "folder",
    "program",
    "file",
    "robot_program.bin",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.801,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 0
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 5,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 5,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "6": {
  "timestep": 6,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 6,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 6,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 6,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 6,
   "action": "router-acl-remove-rule",
   "parameters": {
    "target_router": "edge_router",
    "position": 5
   },
   "request": [
    "network",
    "node",
    "edge_router",
    "acl",
    "remove_rule",
    5
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.8,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 0
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 6,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 6,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "7": {
  "timestep": 7,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 7,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 7,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 7,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 7,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.75,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 0
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 7,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 7,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "8": {
  "timestep": 8,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 8,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "eng_station",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "eng_station",
    "application",
    "web-browser",
    "execute"
   ],
   "response": {
    "status": "failure",
    "data": {}
   },
   "reward": 0.01,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 8,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 8,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 8,
   "action": "node-file-scan",
   "parameters": {
    "node_name": "robot_controller",
    "folder_name": "program",
    "file_name": "robot_program.bin"
   },
   "request": [
    "network",
    "node",
    "robot_controller",
    "file_system",
    "folder",
    "program",
    "file",
    "robot_program.bin",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.801,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 0
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 8,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 8,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "9": {
  "timestep": 9,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 9,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 9,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 9,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 9,
   "action": "node-shutdown",
   "parameters": {
    "node_name": "robot_controller"
   },
   "request": [
    "network",
    "node",
    "robot_controller",
    "shutdown"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.8,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 0
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 9,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 9,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "10": {
  "timestep": 10,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 10,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 10,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 10,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 10,
   "action": "router-acl-remove-rule",
   "parameters": {
    "target_router": "edge_router",
    "position": 5
   },
   "request": [
    "network",
    "node",
    "edge_router",
    "acl",
    "remove_rule",
    5
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.8,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "operating_status": 4,
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 0
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 10,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 10,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "11": {
  "timestep": 11,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 11,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "eng_station",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "eng_station",
    "application",
    "web-browser",
    "execute"
   ],
   "response": {
    "status": "failure",
    "data": {}
   },
   "reward": 0.01,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 11,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "robot_historian",
    "application_name": "database-client"
   },
   "request": [
    "network",
    "node",
    "robot_historian",
    "application",
    "database-client",
    "execute"
   ],
   "response": {
    "status": "failure",
    "data": {}
   },
   "reward": -0.5,
   "reward_info": {
    "connection_attempt_status": "failure"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 11,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 11,
   "action": "node-network-service-recon",
   "parameters": {
    "source_node": "eng_station",
    "target_ip_address": [
     "10.2.20.0/24"
    ],
    "target_port": 22,
    "target_protocol": "tcp",
    "show": true
   },
   "request": [
    "network",
    "node",
    "eng_station",
    "application",
    "nmap",
    "network_service_recon",
    {
     "target_ip_address": [
      "10.2.20.0/24"
     ],
     "target_port": 22,
     "target_protocol": "tcp",
     "show": true
    }
   ],
   "response": {
    "status": "success",
    "data": {
     "10.2.20.30": {
      "tcp": [
       22
      ]
     },
     "10.2.20.20": {
      "tcp": [
       22
      ]
     },
     "10.2.20.1": {
      "tcp": [
       22
      ]
     },
     "10.2.20.40": {
      "tcp": [
       22
      ]
     }
    }
   },
   "reward": 0.801,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "operating_status": 4,
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 0
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 11,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 11,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "12": {
  "timestep": 12,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 12,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "eng_station",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "eng_station",
    "application",
    "web-browser",
    "execute"
   ],
   "response": {
    "status": "failure",
    "data": {}
   },
   "reward": 0.01,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 12,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 12,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "office_pc",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "office_pc",
    "application",
    "web-browser",
    "execute"
   ],
   "response": {
    "status": "failure",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 12,
   "action": "node-application-scan",
   "parameters": {
    "node_name": "robot_controller",
    "application_name": "database-client"
   },
   "request": [
    "network",
    "node",
    "robot_controller",
    "application",
    "database-client",
    "scan"
   ],
   "response": {
    "status": "failure",
    "data": {
     "reason": "Cannot perform request on node 'robot_controller' because it is not powered on."
    }
   },
   "reward": 0.801,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 1,
          "outbound": 1
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 1,
           "outbound": 1
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "operating_status": 4,
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 1,
          "outbound": 1
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 1,
           "outbound": 1
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 1,
          "outbound": 1
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 1,
           "outbound": 1
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 1,
          "outbound": 1
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 1,
           "outbound": 1
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 1
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 12,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 12,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "13": {
  "timestep": 13,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 13,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 13,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 13,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 13,
   "action": "node-file-scan",
   "parameters": {
    "node_name": "robot_controller",
    "folder_name": "program",
    "file_name": "robot_program.bin"
   },
   "request": [
    "network",
    "node",
    "robot_controller",
    "file_system",
    "folder",
    "program",
    "file",
    "robot_program.bin",
    "scan"
   ],
   "response": {
    "status": "failure",
    "data": {
     "reason": "Cannot perform request on node 'robot_controller' because it is not powered on."
    }
   },
   "reward": 0.8,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "operating_status": 2,
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 0
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 13,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 13,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "14": {
  "timestep": 14,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 14,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 14,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 14,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "office_pc",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "office_pc",
    "application",
    "web-browser",
    "execute"
   ],
   "response": {
    "status": "failure",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 14,
   "action": "router-acl-add-rule",
   "parameters": {
    "target_router": "edge_router",
    "permission": "DENY",
    "protocol_name": "tcp",
    "src_ip": "10.2.0.10",
    "src_wildcard": "0.0.0.0",
    "src_port": "ALL",
    "dst_ip": "10.2.20.10",
    "dst_wildcard": "0.0.0.0",
    "dst_port": "SSH",
    "position": 5
   },
   "request": [
    "network",
    "node",
    "edge_router",
    "acl",
    "add_rule",
    "DENY",
    "tcp",
    "10.2.0.10",
    "0.0.0.0",
    "ALL",
    "10.2.20.10",
    "0.0.0.0",
    22,
    5
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.8,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "operating_status": 2,
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 0
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 14,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "vision_pc",
    "application_name": "data-manipulation-bot"
   },
   "request": [
    "network",
    "node",
    "vision_pc",
    "application",
    "data-manipulation-bot",
    "execute"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 14,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "15": {
  "timestep": 15,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 15,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 15,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 15,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "office_pc",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "office_pc",
    "application",
    "web-browser",
    "execute"
   ],
   "response": {
    "status": "failure",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 15,
   "action": "node-file-scan",
   "parameters": {
    "node_name": "robot_controller",
    "folder_name": "program",
    "file_name": "robot_program.bin"
   },
   "request": [
    "network",
    "node",
    "robot_controller",
    "file_system",
    "folder",
    "program",
    "file",
    "robot_program.bin",
    "scan"
   ],
   "response": {
    "status": "failure",
    "data": {
     "reason": "Cannot perform request on node 'robot_controller' because it is not powered on."
    }
   },
   "reward": 0.8,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "operating_status": 2,
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 2,
        "source_wildcard_id": 2,
        "source_port_id": 1,
        "dest_ip_id": 5,
        "dest_wildcard_id": 2,
        "dest_port_id": 3,
        "protocol_id": 3
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 0
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 15,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 15,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "16": {
  "timestep": 16,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 16,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 16,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 16,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "office_pc",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "office_pc",
    "application",
    "web-browser",
    "execute"
   ],
   "response": {
    "status": "failure",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 16,
   "action": "node-network-service-recon",
   "parameters": {
    "source_node": "eng_station",
    "target_ip_address": [
     "10.2.20.0/24"
    ],
    "target_port": 22,
    "target_protocol": "tcp",
    "show": true
   },
   "request": [
    "network",
    "node",
    "eng_station",
    "application",
    "nmap",
    "network_service_recon",
    {
     "target_ip_address": [
      "10.2.20.0/24"
     ],
     "target_port": 22,
     "target_protocol": "tcp",
     "show": true
    }
   ],
   "response": {
    "status": "success",
    "data": {
     "10.2.20.30": {
      "tcp": [
       22
      ]
     },
     "10.2.20.20": {
      "tcp": [
       22
      ]
     },
     "10.2.20.1": {
      "tcp": [
       22
      ]
     },
     "10.2.20.40": {
      "tcp": [
       22
      ]
     }
    }
   },
   "reward": 0.8,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "operating_status": 2,
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 2,
        "source_wildcard_id": 2,
        "source_port_id": 1,
        "dest_ip_id": 5,
        "dest_wildcard_id": 2,
        "dest_port_id": 3,
        "protocol_id": 3
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 0
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 16,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 16,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "17": {
  "timestep": 17,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 17,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 17,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "robot_historian",
    "application_name": "database-client"
   },
   "request": [
    "network",
    "node",
    "robot_historian",
    "application",
    "database-client",
    "execute"
   ],
   "response": {
    "status": "failure",
    "data": {}
   },
   "reward": -0.5,
   "reward_info": {
    "connection_attempt_status": "failure"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 17,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "office_pc",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "office_pc",
    "application",
    "web-browser",
    "execute"
   ],
   "response": {
    "status": "failure",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 17,
   "action": "router-acl-remove-rule",
   "parameters": {
    "target_router": "edge_router",
    "position": 5
   },
   "request": [
    "network",
    "node",
    "edge_router",
    "acl",
    "remove_rule",
    5
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.8,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 1,
          "outbound": 1
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 1,
           "outbound": 1
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "operating_status": 2,
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 1,
          "outbound": 1
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 1,
           "outbound": 1
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 1,
          "outbound": 1
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 1,
           "outbound": 1
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 1,
          "outbound": 1
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 1,
           "outbound": 1
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 2,
        "source_wildcard_id": 2,
        "source_port_id": 1,
        "dest_ip_id": 5,
        "dest_wildcard_id": 2,
        "dest_port_id": 3,
        "protocol_id": 3
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 1
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 17,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 17,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "18": {
  "timestep": 18,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 18,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "eng_station",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "eng_station",
    "application",
    "web-browser",
    "execute"
   ],
   "response": {
    "status": "failure",
    "data": {}
   },
   "reward": 0.01,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 18,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 18,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "office_pc",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "office_pc",
    "application",
    "web-browser",
    "execute"
   ],
   "response": {
    "status": "failure",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 18,
   "action": "node-network-service-recon",
   "parameters": {
    "source_node": "eng_station",
    "target_ip_address": [
     "10.2.20.0/24"
    ],
    "target_port": 22,
    "target_protocol": "tcp",
    "show": true
   },
   "request": [
    "network",
    "node",
    "eng_station",
    "application",
    "nmap",
    "network_service_recon",
    {
     "target_ip_address": [
      "10.2.20.0/24"
     ],
     "target_port": 22,
     "target_protocol": "tcp",
     "show": true
    }
   ],
   "response": {
    "status": "success",
    "data": {
     "10.2.20.30": {
      "tcp": [
       22
      ]
     },
     "10.2.20.20": {
      "tcp": [
       22
      ]
     },
     "10.2.20.1": {
      "tcp": [
       22
      ]
     },
     "10.2.20.40": {
      "tcp": [
       22
      ]
     }
    }
   },
   "reward": 0.801,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "operating_status": 2,
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 0
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 18,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 18,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "19": {
  "timestep": 19,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 19,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 19,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 19,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 19,
   "action": "node-startup",
   "parameters": {
    "node_name": "robot_controller"
   },
   "request": [
    "network",
    "node",
    "robot_controller",
    "startup"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.8,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 1,
          "outbound": 1
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 1,
           "outbound": 1
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "operating_status": 2,
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 1,
          "outbound": 1
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 1,
           "outbound": 1
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 1,
          "outbound": 1
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 1,
           "outbound": 1
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 1,
          "outbound": 1
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 1,
           "outbound": 1
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 1
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 19,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 19,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "20": {
  "timestep": 20,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 20,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "eng_station",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "eng_station",
    "application",
    "web-browser",
    "execute"
   ],
   "response": {
    "status": "failure",
    "data": {}
   },
   "reward": 0.01,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 20,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 20,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "office_pc",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "office_pc",
    "application",
    "web-browser",
    "execute"
   ],
   "response": {
    "status": "failure",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 20,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.751,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "operating_status": 3,
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 0
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 20,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 20,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "21": {
  "timestep": 21,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 21,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "eng_station",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "eng_station",
    "application",
    "web-browser",
    "execute"
   ],
   "response": {
    "status": "failure",
    "data": {}
   },
   "reward": 0.01,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 21,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "robot_historian",
    "application_name": "database-client"
   },
   "request": [
    "network",
    "node",
    "robot_historian",
    "application",
    "database-client",
    "execute"
   ],
   "response": {
    "status": "failure",
    "data": {}
   },
   "reward": -0.5,
   "reward_info": {
    "connection_attempt_status": "failure"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 21,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 21,
   "action": "node-file-scan",
   "parameters": {
    "node_name": "robot_controller",
    "folder_name": "program",
    "file_name": "robot_program.bin"
   },
   "request": [
    "network",
    "node",
    "robot_controller",
    "file_system",
    "folder",
    "program",
    "file",
    "robot_program.bin",
    "scan"
   ],
   "response": {
    "status": "failure",
    "data": {
     "reason": "Cannot perform request on node 'robot_controller' because it is not powered on."
    }
   },
   "reward": 0.801,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "operating_status": 3,
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 0
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 21,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 21,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "22": {
  "timestep": 22,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 22,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 22,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 22,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 22,
   "action": "node-network-service-recon",
   "parameters": {
    "source_node": "eng_station",
    "target_ip_address": [
     "10.2.20.0/24"
    ],
    "target_port": 22,
    "target_protocol": "tcp",
    "show": true
   },
   "request": [
    "network",
    "node",
    "eng_station",
    "application",
    "nmap",
    "network_service_recon",
    {
     "target_ip_address": [
      "10.2.20.0/24"
     ],
     "target_port": 22,
     "target_protocol": "tcp",
     "show": true
    }
   ],
   "response": {
    "status": "success",
    "data": {
     "10.2.20.30": {
      "tcp": [
       22
      ]
     },
     "10.2.20.20": {
      "tcp": [
       22
      ]
     },
     "10.2.20.1": {
      "tcp": [
       22
      ]
     },
     "10.2.20.40": {
      "tcp": [
       22
      ]
     }
    }
   },
   "reward": 0.8,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "operating_status": 3,
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 0
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 22,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 22,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "23": {
  "timestep": 23,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 23,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "eng_station",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "eng_station",
    "application",
    "web-browser",
    "execute"
   ],
   "response": {
    "status": "failure",
    "data": {}
   },
   "reward": 0.01,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 23,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 23,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "office_pc",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "office_pc",
    "application",
    "web-browser",
    "execute"
   ],
   "response": {
    "status": "failure",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 23,
   "action": "router-acl-add-rule",
   "parameters": {
    "target_router": "edge_router",
    "permission": "DENY",
    "protocol_name": "tcp",
    "src_ip": "10.2.0.10",
    "src_wildcard": "0.0.0.0",
    "src_port": "ALL",
    "dst_ip": "10.2.20.10",
    "dst_wildcard": "0.0.0.0",
    "dst_port": "SSH",
    "position": 5
   },
   "request": [
    "network",
    "node",
    "edge_router",
    "acl",
    "add_rule",
    "DENY",
    "tcp",
    "10.2.0.10",
    "0.0.0.0",
    "ALL",
    "10.2.20.10",
    "0.0.0.0",
    22,
    5
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.801,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 1,
          "outbound": 1
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 1,
           "outbound": 1
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 1,
          "outbound": 1
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 1,
           "outbound": 1
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 1,
          "outbound": 1
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 1,
           "outbound": 1
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 1,
          "outbound": 1
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 1,
           "outbound": 1
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 1
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 23,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 23,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "24": {
  "timestep": 24,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 24,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 24,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 24,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 24,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.75,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 2,
        "source_wildcard_id": 2,
        "source_port_id": 1,
        "dest_ip_id": 5,
        "dest_wildcard_id": 2,
        "dest_port_id": 3,
        "protocol_id": 3
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 0
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 24,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 24,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "25": {
  "timestep": 25,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 25,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 25,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 25,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 25,
   "action": "node-file-scan",
   "parameters": {
    "node_name": "robot_controller",
    "folder_name": "program",
    "file_name": "robot_program.bin"
   },
   "request": [
    "network",
    "node",
    "robot_controller",
    "file_system",
    "folder",
    "program",
    "file",
    "robot_program.bin",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.8,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 2,
        "source_wildcard_id": 2,
        "source_port_id": 1,
        "dest_ip_id": 5,
        "dest_wildcard_id": 2,
        "dest_port_id": 3,
        "protocol_id": 3
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 0
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 25,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 25,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "26": {
  "timestep": 26,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 26,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 26,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "robot_historian",
    "application_name": "database-client"
   },
   "request": [
    "network",
    "node",
    "robot_historian",
    "application",
    "database-client",
    "execute"
   ],
   "response": {
    "status": "failure",
    "data": {}
   },
   "reward": -0.5,
   "reward_info": {
    "connection_attempt_status": "failure"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 26,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "office_pc",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "office_pc",
    "application",
    "web-browser",
    "execute"
   ],
   "response": {
    "status": "failure",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 26,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.75,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 2,
        "source_wildcard_id": 2,
        "source_port_id": 1,
        "dest_ip_id": 5,
        "dest_wildcard_id": 2,
        "dest_port_id": 3,
        "protocol_id": 3
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 0
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 26,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 26,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "27": {
  "timestep": 27,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 27,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 27,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 27,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "office_pc",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "office_pc",
    "application",
    "web-browser",
    "execute"
   ],
   "response": {
    "status": "failure",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 27,
   "action": "router-acl-add-rule",
   "parameters": {
    "target_router": "edge_router",
    "permission": "DENY",
    "protocol_name": "tcp",
    "src_ip": "10.2.0.10",
    "src_wildcard": "0.0.0.0",
    "src_port": "ALL",
    "dst_ip": "10.2.20.10",
    "dst_wildcard": "0.0.0.0",
    "dst_port": "SSH",
    "position": 5
   },
   "request": [
    "network",
    "node",
    "edge_router",
    "acl",
    "add_rule",
    "DENY",
    "tcp",
    "10.2.0.10",
    "0.0.0.0",
    "ALL",
    "10.2.20.10",
    "0.0.0.0",
    22,
    5
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.8,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 2,
        "source_wildcard_id": 2,
        "source_port_id": 1,
        "dest_ip_id": 5,
        "dest_wildcard_id": 2,
        "dest_port_id": 3,
        "protocol_id": 3
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 0
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 27,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 27,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "28": {
  "timestep": 28,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 28,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 28,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 28,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "office_pc",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "office_pc",
    "application",
    "web-browser",
    "execute"
   ],
   "response": {
    "status": "failure",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 28,
   "action": "router-acl-remove-rule",
   "parameters": {
    "target_router": "edge_router",
    "position": 5
   },
   "request": [
    "network",
    "node",
    "edge_router",
    "acl",
    "remove_rule",
    5
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.8,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 2,
        "source_wildcard_id": 2,
        "source_port_id": 1,
        "dest_ip_id": 5,
        "dest_wildcard_id": 2,
        "dest_port_id": 3,
        "protocol_id": 3
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 0
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 28,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 28,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 },
 "29": {
  "timestep": 29,
  "episode": 1,
  "OPERATOR_HMI": {
   "timestep": 29,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "HISTORIAN_CLIENT": {
   "timestep": 29,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "SAFETY_ENGINEER": {
   "timestep": 29,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "robotics_guard": {
   "timestep": 29,
   "action": "node-file-scan",
   "parameters": {
    "node_name": "robot_controller",
    "folder_name": "program",
    "file_name": "robot_program.bin"
   },
   "request": [
    "network",
    "node",
    "robot_controller",
    "file_system",
    "folder",
    "program",
    "file",
    "robot_program.bin",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.8,
   "reward_info": {},
   "observation": {
    "NODES": {
     "HOST0": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST1": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST2": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST3": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST4": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST5": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "HOST6": {
      "SERVICES": {
       "1": {
        "operating_status": 0,
        "health_status": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0
       }
      },
      "APPLICATIONS": {
       "1": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       },
       "2": {
        "operating_status": 0,
        "health_status": 0,
        "num_executions": 0
       }
      },
      "FOLDERS": {
       "1": {
        "health_status": 0,
        "FILES": {
         "1": {
          "health_status": 0
         }
        }
       }
      },
      "NICS": {
       "1": {
        "nic_status": 1,
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        },
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        }
       },
       "2": {
        "nic_status": 0,
        "NMNE": {
         "inbound": 0,
         "outbound": 0
        },
        "TRAFFIC": {
         "icmp": {
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "22": {
           "inbound": 0,
           "outbound": 0
          }
         },
         "udp": {
          "0": {
           "inbound": 0,
           "outbound": 0
          }
         }
        }
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      },
      "operating_status": 1
     },
     "ROUTER0": {
      "ACL": {
       "0": {
        "position": 0,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "1": {
        "position": 1,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 1,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 1,
        "protocol_id": 2
       },
       "2": {
        "position": 2,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 2,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 2,
        "protocol_id": 1
       },
       "3": {
        "position": 3,
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 3,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 3,
        "protocol_id": 1
       },
       "4": {
        "position": 4,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "5": {
        "position": 5,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "6": {
        "position": 6,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "7": {
        "position": 7,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "8": {
        "position": 8,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "9": {
        "position": 9,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "10": {
        "position": 10,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       },
       "11": {
        "position": 11,
        "permission": 0,
        "source_ip_id": 0,
        "source_wildcard_id": 0,
        "source_port_id": 0,
        "dest_ip_id": 0,
        "dest_wildcard_id": 0,
        "dest_port_id": 0,
        "protocol_id": 0
       }
      },
      "PORTS": {
       "1": {
        "operating_status": 1
       },
       "2": {
        "operating_status": 1
       }
      },
      "users": {
       "local_login": 0,
       "remote_sessions": 0
      }
     }
    },
    "LINKS": {
     "1": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "2": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "3": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "4": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "5": {
      "PROTOCOLS": {
       "ALL": 1
      }
     },
     "6": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "7": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "8": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 0
      }
     },
     "10": {
      "PROTOCOLS": {
       "ALL": 0
      }
     }
    }
   }
  },
  "robotics_attacker": {
   "timestep": 29,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "vision_pc",
    "application_name": "data-manipulation-bot"
   },
   "request": [
    "network",
    "node",
    "vision_pc",
    "application",
    "data-manipulation-bot",
    "execute"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  },
  "apt_master": {
   "timestep": 29,
   "action": "do-nothing",
   "parameters": {},
   "request": [
    "do-nothing"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.0,
   "reward_info": {},
   "observation": 0
  }
 }
}