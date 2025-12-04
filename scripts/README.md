{
 "0": {
  "timestep": 0,
  "episode": 1,
  "campus_user": {
   "timestep": 0,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
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
  "ENG_HISTORIAN_USER": {
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
  "HMI_OPERATOR": {
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
  "BACKUP_VALIDATOR": {
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
   "reward": 0.3,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 0,
   "action": "node-file-repair",
   "parameters": {
    "node_name": "ot_controller",
    "folder_name": "database",
    "file_name": "control.db"
   },
   "request": [
    "network",
    "node",
    "ot_controller",
    "file_system",
    "folder",
    "database",
    "file",
    "control.db",
    "repair"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.8800000000000001,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
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
     }
    }
   }
  },
  "lateral_mover": {
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
  "dos_attacker": {
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
  "ransomware_attacker": {
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
  "campus_user": {
   "timestep": 1,
   "action": "node-application-scan",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
    "application",
    "web-browser",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.01,
   "reward_info": {},
   "observation": 0
  },
  "ENG_HISTORIAN_USER": {
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
  "HMI_OPERATOR": {
   "timestep": 1,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
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
  "BACKUP_VALIDATOR": {
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
   "reward": 0.3,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 1,
   "action": "node-file-scan",
   "parameters": {
    "node_name": "ot_controller",
    "folder_name": "database",
    "file_name": "control.db"
   },
   "request": [
    "network",
    "node",
    "ot_controller",
    "file_system",
    "folder",
    "database",
    "file",
    "control.db",
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
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 1,
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
         "inbound": 1,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
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
     }
    }
   }
  },
  "lateral_mover": {
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
  "dos_attacker": {
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
  "ransomware_attacker": {
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
  "campus_user": {
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
  "ENG_HISTORIAN_USER": {
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
  "HMI_OPERATOR": {
   "timestep": 2,
   "action": "node-application-scan",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
    "application",
    "web-browser",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "BACKUP_VALIDATOR": {
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
   "reward": 0.3,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 2,
   "action": "node-application-scan",
   "parameters": {
    "node_name": "ot_gateway",
    "application_name": "database-client"
   },
   "request": [
    "network",
    "node",
    "ot_gateway",
    "application",
    "database-client",
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
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
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
     }
    }
   }
  },
  "lateral_mover": {
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
  "dos_attacker": {
   "timestep": 2,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "dmz_proxy",
    "application_name": "dos-bot"
   },
   "request": [
    "network",
    "node",
    "dmz_proxy",
    "application",
    "dos-bot",
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
  "ransomware_attacker": {
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
  "campus_user": {
   "timestep": 3,
   "action": "node-application-scan",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
    "application",
    "web-browser",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.01,
   "reward_info": {},
   "observation": 0
  },
  "ENG_HISTORIAN_USER": {
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
  "HMI_OPERATOR": {
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
  "BACKUP_VALIDATOR": {
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
   "reward": 0.3,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 3,
   "action": "node-network-service-recon",
   "parameters": {
    "source_node": "security_monitor",
    "target_ip_address": [
     "10.0.20.0/24"
    ],
    "target_port": 502,
    "target_protocol": "tcp",
    "show": true
   },
   "request": [
    "network",
    "node",
    "security_monitor",
    "application",
    "nmap",
    "network_service_recon",
    {
     "target_ip_address": [
      "10.0.20.0/24"
     ],
     "target_port": 502,
     "target_protocol": "tcp",
     "show": true
    }
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
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
     }
    }
   }
  },
  "lateral_mover": {
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
  "dos_attacker": {
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
  "ransomware_attacker": {
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
  "campus_user": {
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
  "ENG_HISTORIAN_USER": {
   "timestep": 4,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "ot_gateway",
    "application_name": "database-client"
   },
   "request": [
    "network",
    "node",
    "ot_gateway",
    "application",
    "database-client",
    "execute"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.5,
   "reward_info": {
    "connection_attempt_status": "success"
   },
   "observation": 0
  },
  "HMI_OPERATOR": {
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
  "BACKUP_VALIDATOR": {
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
   "reward": 0.3,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 4,
   "action": "node-network-service-recon",
   "parameters": {
    "source_node": "security_monitor",
    "target_ip_address": [
     "10.0.20.0/24"
    ],
    "target_port": 502,
    "target_protocol": "tcp",
    "show": true
   },
   "request": [
    "network",
    "node",
    "security_monitor",
    "application",
    "nmap",
    "network_service_recon",
    {
     "target_ip_address": [
      "10.0.20.0/24"
     ],
     "target_port": 502,
     "target_protocol": "tcp",
     "show": true
    }
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.9000000000000001,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "inbound": 1,
          "outbound": 1
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
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
     }
    }
   }
  },
  "lateral_mover": {
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
  "dos_attacker": {
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
  "ransomware_attacker": {
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
   "action": "node-application-execute",
   "parameters": {
    "node_name": "dmz_proxy",
    "application_name": "data-manipulation-bot"
   },
   "request": [
    "network",
    "node",
    "dmz_proxy",
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
  }
 },
 "5": {
  "timestep": 5,
  "episode": 1,
  "campus_user": {
   "timestep": 5,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
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
  "ENG_HISTORIAN_USER": {
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
   "reward": 0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "HMI_OPERATOR": {
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
  "BACKUP_VALIDATOR": {
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
   "reward": 0.3,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 5,
   "action": "router-acl-add-rule",
   "parameters": {
    "target_router": "edge_router",
    "permission": "DENY",
    "protocol_name": "tcp",
    "src_ip": "10.0.20.1",
    "src_wildcard": "0.0.0.255",
    "src_port": "ALL",
    "dst_ip": "10.0.0.20",
    "dst_wildcard": "0.0.0.0",
    "dst_port": "ALL",
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
    "10.0.20.1",
    "0.0.0.255",
    "ALL",
    "10.0.0.20",
    "0.0.0.0",
    "ALL",
    5
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.9000000000000001,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "inbound": 1,
          "outbound": 1
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
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
     }
    }
   }
  },
  "lateral_mover": {
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
  "dos_attacker": {
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
  "ransomware_attacker": {
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
  "campus_user": {
   "timestep": 6,
   "action": "node-application-scan",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
    "application",
    "web-browser",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.01,
   "reward_info": {},
   "observation": 0
  },
  "ENG_HISTORIAN_USER": {
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
   "reward": 0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "HMI_OPERATOR": {
   "timestep": 6,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
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
  "BACKUP_VALIDATOR": {
   "timestep": 6,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "ot_controller",
    "application_name": "database-service"
   },
   "request": [
    "network",
    "node",
    "ot_controller",
    "application",
    "database-service",
    "execute"
   ],
   "response": {
    "status": "unreachable",
    "data": {
     "reason": [
      "Request ['database-service', 'execute'] could not be processed because database-service is not a valid request name",
      "within this RequestManager"
     ]
    }
   },
   "reward": 0.31,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 6,
   "action": "router-acl-add-rule",
   "parameters": {
    "target_router": "edge_router",
    "permission": "DENY",
    "protocol_name": "tcp",
    "src_ip": "0.0.0.0",
    "src_wildcard": "0.0.0.0",
    "src_port": "ALL",
    "dst_ip": "10.0.20.0",
    "dst_wildcard": "0.0.0.255",
    "dst_port": "ALL",
    "position": 6
   },
   "request": [
    "network",
    "node",
    "edge_router",
    "acl",
    "add_rule",
    "DENY",
    "tcp",
    "0.0.0.0",
    "0.0.0.0",
    "ALL",
    "10.0.20.0",
    "0.0.0.255",
    "ALL",
    6
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.9010000000000001,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 5,
        "source_wildcard_id": 3,
        "source_port_id": 1,
        "dest_ip_id": 2,
        "dest_wildcard_id": 2,
        "dest_port_id": 1,
        "protocol_id": 3
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
     }
    }
   }
  },
  "lateral_mover": {
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
  "dos_attacker": {
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
  "ransomware_attacker": {
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
  "campus_user": {
   "timestep": 7,
   "action": "node-application-scan",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
    "application",
    "web-browser",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.01,
   "reward_info": {},
   "observation": 0
  },
  "ENG_HISTORIAN_USER": {
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
   "reward": 0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "HMI_OPERATOR": {
   "timestep": 7,
   "action": "node-application-scan",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
    "application",
    "web-browser",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "BACKUP_VALIDATOR": {
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
   "reward": 0.3,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 7,
   "action": "node-file-repair",
   "parameters": {
    "node_name": "ot_controller",
    "folder_name": "database",
    "file_name": "control.db"
   },
   "request": [
    "network",
    "node",
    "ot_controller",
    "file_system",
    "folder",
    "database",
    "file",
    "control.db",
    "repair"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.9000000000000001,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 5,
        "source_wildcard_id": 3,
        "source_port_id": 1,
        "dest_ip_id": 2,
        "dest_wildcard_id": 2,
        "dest_port_id": 1,
        "protocol_id": 3
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
     }
    }
   }
  },
  "lateral_mover": {
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
  "dos_attacker": {
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
  "ransomware_attacker": {
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
  "campus_user": {
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
  "ENG_HISTORIAN_USER": {
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
   "reward": 0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "HMI_OPERATOR": {
   "timestep": 8,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
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
  "BACKUP_VALIDATOR": {
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
   "reward": 0.3,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 8,
   "action": "node-application-scan",
   "parameters": {
    "node_name": "ot_gateway",
    "application_name": "database-client"
   },
   "request": [
    "network",
    "node",
    "ot_gateway",
    "application",
    "database-client",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.9000000000000001,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 5,
        "source_wildcard_id": 3,
        "source_port_id": 1,
        "dest_ip_id": 2,
        "dest_wildcard_id": 2,
        "dest_port_id": 1,
        "protocol_id": 3
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
     }
    }
   }
  },
  "lateral_mover": {
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
  "dos_attacker": {
   "timestep": 8,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "dmz_proxy",
    "application_name": "dos-bot"
   },
   "request": [
    "network",
    "node",
    "dmz_proxy",
    "application",
    "dos-bot",
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
  "ransomware_attacker": {
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
  "campus_user": {
   "timestep": 9,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
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
  "ENG_HISTORIAN_USER": {
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
   "reward": 0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "HMI_OPERATOR": {
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
  "BACKUP_VALIDATOR": {
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
   "reward": 0.3,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 9,
   "action": "node-network-service-recon",
   "parameters": {
    "source_node": "security_monitor",
    "target_ip_address": [
     "10.0.20.0/24"
    ],
    "target_port": 502,
    "target_protocol": "tcp",
    "show": true
   },
   "request": [
    "network",
    "node",
    "security_monitor",
    "application",
    "nmap",
    "network_service_recon",
    {
     "target_ip_address": [
      "10.0.20.0/24"
     ],
     "target_port": 502,
     "target_protocol": "tcp",
     "show": true
    }
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.9000000000000001,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 5,
        "source_wildcard_id": 3,
        "source_port_id": 1,
        "dest_ip_id": 2,
        "dest_wildcard_id": 2,
        "dest_port_id": 1,
        "protocol_id": 3
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
     }
    }
   }
  },
  "lateral_mover": {
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
  "dos_attacker": {
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
  "ransomware_attacker": {
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
   "action": "node-application-execute",
   "parameters": {
    "node_name": "dmz_proxy",
    "application_name": "data-manipulation-bot"
   },
   "request": [
    "network",
    "node",
    "dmz_proxy",
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
  }
 },
 "10": {
  "timestep": 10,
  "episode": 1,
  "campus_user": {
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
  "ENG_HISTORIAN_USER": {
   "timestep": 10,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "ot_gateway",
    "application_name": "database-client"
   },
   "request": [
    "network",
    "node",
    "ot_gateway",
    "application",
    "database-client",
    "execute"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.5,
   "reward_info": {
    "connection_attempt_status": "success"
   },
   "observation": 0
  },
  "HMI_OPERATOR": {
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
  "BACKUP_VALIDATOR": {
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
   "reward": 0.3,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 10,
   "action": "router-acl-add-rule",
   "parameters": {
    "target_router": "edge_router",
    "permission": "DENY",
    "protocol_name": "tcp",
    "src_ip": "10.0.20.1",
    "src_wildcard": "0.0.0.255",
    "src_port": "ALL",
    "dst_ip": "10.0.0.20",
    "dst_wildcard": "0.0.0.0",
    "dst_port": "ALL",
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
    "10.0.20.1",
    "0.0.0.255",
    "ALL",
    "10.0.0.20",
    "0.0.0.0",
    "ALL",
    5
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.9000000000000001,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "inbound": 1,
          "outbound": 1
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 5,
        "source_wildcard_id": 3,
        "source_port_id": 1,
        "dest_ip_id": 2,
        "dest_wildcard_id": 2,
        "dest_port_id": 1,
        "protocol_id": 3
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
     }
    }
   }
  },
  "lateral_mover": {
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
  "dos_attacker": {
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
  "ransomware_attacker": {
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
  "campus_user": {
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
  "ENG_HISTORIAN_USER": {
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
   "reward": 0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "HMI_OPERATOR": {
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
  "BACKUP_VALIDATOR": {
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
   "reward": 0.3,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 11,
   "action": "router-acl-add-rule",
   "parameters": {
    "target_router": "edge_router",
    "permission": "DENY",
    "protocol_name": "tcp",
    "src_ip": "10.0.20.1",
    "src_wildcard": "0.0.0.255",
    "src_port": "ALL",
    "dst_ip": "10.0.0.20",
    "dst_wildcard": "0.0.0.0",
    "dst_port": "ALL",
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
    "10.0.20.1",
    "0.0.0.255",
    "ALL",
    "10.0.0.20",
    "0.0.0.0",
    "ALL",
    5
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.9000000000000001,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 5,
        "source_wildcard_id": 3,
        "source_port_id": 1,
        "dest_ip_id": 2,
        "dest_wildcard_id": 2,
        "dest_port_id": 1,
        "protocol_id": 3
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
       "ALL": 0
      }
     }
    }
   }
  },
  "lateral_mover": {
   "timestep": 11,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "dmz_proxy",
    "application_name": "data-manipulation-bot"
   },
   "request": [
    "network",
    "node",
    "dmz_proxy",
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
  "dos_attacker": {
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
  "ransomware_attacker": {
   "timestep": 11,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "dmz_proxy",
    "application_name": "ransomware-script"
   },
   "request": [
    "network",
    "node",
    "dmz_proxy",
    "application",
    "ransomware-script",
    "execute"
   ],
   "response": {
    "status": "failure",
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
  "campus_user": {
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
  "ENG_HISTORIAN_USER": {
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
   "reward": 0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "HMI_OPERATOR": {
   "timestep": 12,
   "action": "node-application-scan",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
    "application",
    "web-browser",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "BACKUP_VALIDATOR": {
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
   "reward": 0.3,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 12,
   "action": "node-file-repair",
   "parameters": {
    "node_name": "ot_controller",
    "folder_name": "database",
    "file_name": "control.db"
   },
   "request": [
    "network",
    "node",
    "ot_controller",
    "file_system",
    "folder",
    "database",
    "file",
    "control.db",
    "repair"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.9000000000000001,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 5,
        "source_wildcard_id": 3,
        "source_port_id": 1,
        "dest_ip_id": 2,
        "dest_wildcard_id": 2,
        "dest_port_id": 1,
        "protocol_id": 3
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
     }
    }
   }
  },
  "lateral_mover": {
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
  "dos_attacker": {
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
  "ransomware_attacker": {
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
  "campus_user": {
   "timestep": 13,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
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
  "ENG_HISTORIAN_USER": {
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
   "reward": 0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "HMI_OPERATOR": {
   "timestep": 13,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
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
  "BACKUP_VALIDATOR": {
   "timestep": 13,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "ot_controller",
    "application_name": "database-service"
   },
   "request": [
    "network",
    "node",
    "ot_controller",
    "application",
    "database-service",
    "execute"
   ],
   "response": {
    "status": "unreachable",
    "data": {
     "reason": [
      "Request ['database-service', 'execute'] could not be processed because database-service is not a valid request name",
      "within this RequestManager"
     ]
    }
   },
   "reward": 0.31,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 13,
   "action": "node-file-repair",
   "parameters": {
    "node_name": "ot_controller",
    "folder_name": "database",
    "file_name": "control.db"
   },
   "request": [
    "network",
    "node",
    "ot_controller",
    "file_system",
    "folder",
    "database",
    "file",
    "control.db",
    "repair"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.9010000000000001,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 5,
        "source_wildcard_id": 3,
        "source_port_id": 1,
        "dest_ip_id": 2,
        "dest_wildcard_id": 2,
        "dest_port_id": 1,
        "protocol_id": 3
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
     }
    }
   }
  },
  "lateral_mover": {
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
  "dos_attacker": {
   "timestep": 13,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "dmz_proxy",
    "application_name": "dos-bot"
   },
   "request": [
    "network",
    "node",
    "dmz_proxy",
    "application",
    "dos-bot",
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
  "ransomware_attacker": {
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
  "campus_user": {
   "timestep": 14,
   "action": "node-application-scan",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
    "application",
    "web-browser",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.01,
   "reward_info": {},
   "observation": 0
  },
  "ENG_HISTORIAN_USER": {
   "timestep": 14,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "ot_gateway",
    "application_name": "database-client"
   },
   "request": [
    "network",
    "node",
    "ot_gateway",
    "application",
    "database-client",
    "execute"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.5,
   "reward_info": {
    "connection_attempt_status": "success"
   },
   "observation": 0
  },
  "HMI_OPERATOR": {
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
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "BACKUP_VALIDATOR": {
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
   "reward": 0.3,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 14,
   "action": "router-acl-add-rule",
   "parameters": {
    "target_router": "edge_router",
    "permission": "DENY",
    "protocol_name": "tcp",
    "src_ip": "0.0.0.0",
    "src_wildcard": "0.0.0.0",
    "src_port": "ALL",
    "dst_ip": "10.0.20.0",
    "dst_wildcard": "0.0.0.255",
    "dst_port": "ALL",
    "position": 6
   },
   "request": [
    "network",
    "node",
    "edge_router",
    "acl",
    "add_rule",
    "DENY",
    "tcp",
    "0.0.0.0",
    "0.0.0.0",
    "ALL",
    "10.0.20.0",
    "0.0.0.255",
    "ALL",
    6
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.9000000000000001,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 5,
        "source_wildcard_id": 3,
        "source_port_id": 1,
        "dest_ip_id": 2,
        "dest_wildcard_id": 2,
        "dest_port_id": 1,
        "protocol_id": 3
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
     }
    }
   }
  },
  "lateral_mover": {
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
  "dos_attacker": {
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
  "ransomware_attacker": {
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
  "apt_master": {
   "timestep": 14,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "dmz_proxy",
    "application_name": "data-manipulation-bot"
   },
   "request": [
    "network",
    "node",
    "dmz_proxy",
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
  }
 },
 "15": {
  "timestep": 15,
  "episode": 1,
  "campus_user": {
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
  "ENG_HISTORIAN_USER": {
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
   "reward": 0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "HMI_OPERATOR": {
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
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "BACKUP_VALIDATOR": {
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
   "reward": 0.3,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 15,
   "action": "node-service-fix",
   "parameters": {
    "node_name": "ot_controller",
    "service_name": "database-service"
   },
   "request": [
    "network",
    "node",
    "ot_controller",
    "service",
    "database-service",
    "fix"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.9000000000000001,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 5,
        "source_wildcard_id": 3,
        "source_port_id": 1,
        "dest_ip_id": 2,
        "dest_wildcard_id": 2,
        "dest_port_id": 1,
        "protocol_id": 3
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
       "ALL": 0
      }
     }
    }
   }
  },
  "lateral_mover": {
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
  "dos_attacker": {
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
  "ransomware_attacker": {
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
  "campus_user": {
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
  "ENG_HISTORIAN_USER": {
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
   "reward": 0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "HMI_OPERATOR": {
   "timestep": 16,
   "action": "node-application-scan",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
    "application",
    "web-browser",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "BACKUP_VALIDATOR": {
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
   "reward": 0.3,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 5,
        "source_wildcard_id": 3,
        "source_port_id": 1,
        "dest_ip_id": 2,
        "dest_wildcard_id": 2,
        "dest_port_id": 1,
        "protocol_id": 3
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
     }
    }
   }
  },
  "lateral_mover": {
   "timestep": 16,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "dmz_proxy",
    "application_name": "data-manipulation-bot"
   },
   "request": [
    "network",
    "node",
    "dmz_proxy",
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
  "dos_attacker": {
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
  "ransomware_attacker": {
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
  "campus_user": {
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
  "ENG_HISTORIAN_USER": {
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
   "reward": 0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "HMI_OPERATOR": {
   "timestep": 17,
   "action": "node-application-scan",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
    "application",
    "web-browser",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "BACKUP_VALIDATOR": {
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
   "reward": 0.3,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 1,
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
         "inbound": 1,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 5,
        "source_wildcard_id": 3,
        "source_port_id": 1,
        "dest_ip_id": 2,
        "dest_wildcard_id": 2,
        "dest_port_id": 1,
        "protocol_id": 3
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
       "ALL": 1
      }
     },
     "9": {
      "PROTOCOLS": {
       "ALL": 1
      }
     }
    }
   }
  },
  "lateral_mover": {
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
  "dos_attacker": {
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
  "ransomware_attacker": {
   "timestep": 17,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "dmz_proxy",
    "application_name": "ransomware-script"
   },
   "request": [
    "network",
    "node",
    "dmz_proxy",
    "application",
    "ransomware-script",
    "execute"
   ],
   "response": {
    "status": "failure",
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
  "campus_user": {
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
  "ENG_HISTORIAN_USER": {
   "timestep": 18,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "ot_gateway",
    "application_name": "database-client"
   },
   "request": [
    "network",
    "node",
    "ot_gateway",
    "application",
    "database-client",
    "execute"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.5,
   "reward_info": {
    "connection_attempt_status": "success"
   },
   "observation": 0
  },
  "HMI_OPERATOR": {
   "timestep": 18,
   "action": "node-application-scan",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
    "application",
    "web-browser",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "BACKUP_VALIDATOR": {
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
   "reward": 0.3,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 18,
   "action": "node-application-scan",
   "parameters": {
    "node_name": "ot_gateway",
    "application_name": "database-client"
   },
   "request": [
    "network",
    "node",
    "ot_gateway",
    "application",
    "database-client",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.9000000000000001,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 5,
        "source_wildcard_id": 3,
        "source_port_id": 1,
        "dest_ip_id": 2,
        "dest_wildcard_id": 2,
        "dest_port_id": 1,
        "protocol_id": 3
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
     }
    }
   }
  },
  "lateral_mover": {
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
  "dos_attacker": {
   "timestep": 18,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "dmz_proxy",
    "application_name": "dos-bot"
   },
   "request": [
    "network",
    "node",
    "dmz_proxy",
    "application",
    "dos-bot",
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
  "ransomware_attacker": {
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
  "campus_user": {
   "timestep": 19,
   "action": "node-application-scan",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
    "application",
    "web-browser",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.01,
   "reward_info": {},
   "observation": 0
  },
  "ENG_HISTORIAN_USER": {
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
   "reward": 0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "HMI_OPERATOR": {
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
  "BACKUP_VALIDATOR": {
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
   "reward": 0.3,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 19,
   "action": "node-application-scan",
   "parameters": {
    "node_name": "ot_gateway",
    "application_name": "database-client"
   },
   "request": [
    "network",
    "node",
    "ot_gateway",
    "application",
    "database-client",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.9000000000000001,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 5,
        "source_wildcard_id": 3,
        "source_port_id": 1,
        "dest_ip_id": 2,
        "dest_wildcard_id": 2,
        "dest_port_id": 1,
        "protocol_id": 3
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
       "ALL": 0
      }
     }
    }
   }
  },
  "lateral_mover": {
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
  "dos_attacker": {
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
  "ransomware_attacker": {
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
   "action": "node-application-execute",
   "parameters": {
    "node_name": "dmz_proxy",
    "application_name": "data-manipulation-bot"
   },
   "request": [
    "network",
    "node",
    "dmz_proxy",
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
  }
 },
 "20": {
  "timestep": 20,
  "episode": 1,
  "campus_user": {
   "timestep": 20,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
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
  "ENG_HISTORIAN_USER": {
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
   "reward": 0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "HMI_OPERATOR": {
   "timestep": 20,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
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
  "BACKUP_VALIDATOR": {
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
   "reward": 0.3,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 20,
   "action": "node-file-scan",
   "parameters": {
    "node_name": "ot_controller",
    "folder_name": "database",
    "file_name": "control.db"
   },
   "request": [
    "network",
    "node",
    "ot_controller",
    "file_system",
    "folder",
    "database",
    "file",
    "control.db",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.9000000000000001,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 5,
        "source_wildcard_id": 3,
        "source_port_id": 1,
        "dest_ip_id": 2,
        "dest_wildcard_id": 2,
        "dest_port_id": 1,
        "protocol_id": 3
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
     }
    }
   }
  },
  "lateral_mover": {
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
  "dos_attacker": {
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
  "ransomware_attacker": {
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
  "campus_user": {
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
  "ENG_HISTORIAN_USER": {
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
   "reward": 0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "HMI_OPERATOR": {
   "timestep": 21,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
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
  "BACKUP_VALIDATOR": {
   "timestep": 21,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "ot_controller",
    "application_name": "database-service"
   },
   "request": [
    "network",
    "node",
    "ot_controller",
    "application",
    "database-service",
    "execute"
   ],
   "response": {
    "status": "unreachable",
    "data": {
     "reason": [
      "Request ['database-service', 'execute'] could not be processed because database-service is not a valid request name",
      "within this RequestManager"
     ]
    }
   },
   "reward": 0.31,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 5,
        "source_wildcard_id": 3,
        "source_port_id": 1,
        "dest_ip_id": 2,
        "dest_wildcard_id": 2,
        "dest_port_id": 1,
        "protocol_id": 3
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
     }
    }
   }
  },
  "lateral_mover": {
   "timestep": 21,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "dmz_proxy",
    "application_name": "data-manipulation-bot"
   },
   "request": [
    "network",
    "node",
    "dmz_proxy",
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
  "dos_attacker": {
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
  "ransomware_attacker": {
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
  "campus_user": {
   "timestep": 22,
   "action": "node-application-scan",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
    "application",
    "web-browser",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.01,
   "reward_info": {},
   "observation": 0
  },
  "ENG_HISTORIAN_USER": {
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
   "reward": 0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "HMI_OPERATOR": {
   "timestep": 22,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
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
  "BACKUP_VALIDATOR": {
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
   "reward": 0.3,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 22,
   "action": "node-service-fix",
   "parameters": {
    "node_name": "ot_controller",
    "service_name": "database-service"
   },
   "request": [
    "network",
    "node",
    "ot_controller",
    "service",
    "database-service",
    "fix"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.9000000000000001,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 5,
        "source_wildcard_id": 3,
        "source_port_id": 1,
        "dest_ip_id": 2,
        "dest_wildcard_id": 2,
        "dest_port_id": 1,
        "protocol_id": 3
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
     }
    }
   }
  },
  "lateral_mover": {
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
  "dos_attacker": {
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
  "ransomware_attacker": {
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
   "action": "node-application-execute",
   "parameters": {
    "node_name": "dmz_proxy",
    "application_name": "data-manipulation-bot"
   },
   "request": [
    "network",
    "node",
    "dmz_proxy",
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
  }
 },
 "23": {
  "timestep": 23,
  "episode": 1,
  "campus_user": {
   "timestep": 23,
   "action": "node-application-scan",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
    "application",
    "web-browser",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.01,
   "reward_info": {},
   "observation": 0
  },
  "ENG_HISTORIAN_USER": {
   "timestep": 23,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "ot_gateway",
    "application_name": "database-client"
   },
   "request": [
    "network",
    "node",
    "ot_gateway",
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
  "HMI_OPERATOR": {
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
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "BACKUP_VALIDATOR": {
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
   "reward": 0.3,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 23,
   "action": "node-service-fix",
   "parameters": {
    "node_name": "ot_controller",
    "service_name": "database-service"
   },
   "request": [
    "network",
    "node",
    "ot_controller",
    "service",
    "database-service",
    "fix"
   ],
   "response": {
    "status": "failure",
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 5,
        "source_wildcard_id": 3,
        "source_port_id": 1,
        "dest_ip_id": 2,
        "dest_wildcard_id": 2,
        "dest_port_id": 1,
        "protocol_id": 3
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
     }
    }
   }
  },
  "lateral_mover": {
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
  "dos_attacker": {
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
  "ransomware_attacker": {
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
  "campus_user": {
   "timestep": 24,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
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
  "ENG_HISTORIAN_USER": {
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
  "HMI_OPERATOR": {
   "timestep": 24,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
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
  "BACKUP_VALIDATOR": {
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
   "reward": 0.3,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 24,
   "action": "node-file-scan",
   "parameters": {
    "node_name": "ot_controller",
    "folder_name": "database",
    "file_name": "control.db"
   },
   "request": [
    "network",
    "node",
    "ot_controller",
    "file_system",
    "folder",
    "database",
    "file",
    "control.db",
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 1,
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
         "inbound": 1,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 5,
        "source_wildcard_id": 3,
        "source_port_id": 1,
        "dest_ip_id": 2,
        "dest_wildcard_id": 2,
        "dest_port_id": 1,
        "protocol_id": 3
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
     }
    }
   }
  },
  "lateral_mover": {
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
  "dos_attacker": {
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
  "ransomware_attacker": {
   "timestep": 24,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "dmz_proxy",
    "application_name": "ransomware-script"
   },
   "request": [
    "network",
    "node",
    "dmz_proxy",
    "application",
    "ransomware-script",
    "execute"
   ],
   "response": {
    "status": "failure",
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
  "campus_user": {
   "timestep": 25,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
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
  "ENG_HISTORIAN_USER": {
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
  "HMI_OPERATOR": {
   "timestep": 25,
   "action": "node-application-scan",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
    "application",
    "web-browser",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "BACKUP_VALIDATOR": {
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
   "reward": 0.3,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 25,
   "action": "router-acl-add-rule",
   "parameters": {
    "target_router": "edge_router",
    "permission": "DENY",
    "protocol_name": "tcp",
    "src_ip": "10.0.20.1",
    "src_wildcard": "0.0.0.255",
    "src_port": "ALL",
    "dst_ip": "10.0.0.20",
    "dst_wildcard": "0.0.0.0",
    "dst_port": "ALL",
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
    "10.0.20.1",
    "0.0.0.255",
    "ALL",
    "10.0.0.20",
    "0.0.0.0",
    "ALL",
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 5,
        "source_wildcard_id": 3,
        "source_port_id": 1,
        "dest_ip_id": 2,
        "dest_wildcard_id": 2,
        "dest_port_id": 1,
        "protocol_id": 3
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
     }
    }
   }
  },
  "lateral_mover": {
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
  "dos_attacker": {
   "timestep": 25,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "dmz_proxy",
    "application_name": "dos-bot"
   },
   "request": [
    "network",
    "node",
    "dmz_proxy",
    "application",
    "dos-bot",
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
  "ransomware_attacker": {
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
  "campus_user": {
   "timestep": 26,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
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
  "ENG_HISTORIAN_USER": {
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
   "reward": -0.5,
   "reward_info": {
    "connection_attempt_status": "n/a"
   },
   "observation": 0
  },
  "HMI_OPERATOR": {
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
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "BACKUP_VALIDATOR": {
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
   "reward": 0.3,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 26,
   "action": "node-network-service-recon",
   "parameters": {
    "source_node": "security_monitor",
    "target_ip_address": [
     "10.0.20.0/24"
    ],
    "target_port": 502,
    "target_protocol": "tcp",
    "show": true
   },
   "request": [
    "network",
    "node",
    "security_monitor",
    "application",
    "nmap",
    "network_service_recon",
    {
     "target_ip_address": [
      "10.0.20.0/24"
     ],
     "target_port": 502,
     "target_protocol": "tcp",
     "show": true
    }
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 5,
        "source_wildcard_id": 3,
        "source_port_id": 1,
        "dest_ip_id": 2,
        "dest_wildcard_id": 2,
        "dest_port_id": 1,
        "protocol_id": 3
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
     }
    }
   }
  },
  "lateral_mover": {
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
  "dos_attacker": {
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
  "ransomware_attacker": {
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
  "campus_user": {
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
  "ENG_HISTORIAN_USER": {
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
  "HMI_OPERATOR": {
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
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "BACKUP_VALIDATOR": {
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
   "reward": 0.3,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 27,
   "action": "node-service-fix",
   "parameters": {
    "node_name": "ot_controller",
    "service_name": "database-service"
   },
   "request": [
    "network",
    "node",
    "ot_controller",
    "service",
    "database-service",
    "fix"
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "inbound": 1,
          "outbound": 1
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 5,
        "source_wildcard_id": 3,
        "source_port_id": 1,
        "dest_ip_id": 2,
        "dest_wildcard_id": 2,
        "dest_port_id": 1,
        "protocol_id": 3
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
     }
    }
   }
  },
  "lateral_mover": {
   "timestep": 27,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "dmz_proxy",
    "application_name": "data-manipulation-bot"
   },
   "request": [
    "network",
    "node",
    "dmz_proxy",
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
  "dos_attacker": {
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
  "ransomware_attacker": {
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
   "action": "node-application-execute",
   "parameters": {
    "node_name": "dmz_proxy",
    "application_name": "data-manipulation-bot"
   },
   "request": [
    "network",
    "node",
    "dmz_proxy",
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
  }
 },
 "28": {
  "timestep": 28,
  "episode": 1,
  "campus_user": {
   "timestep": 28,
   "action": "node-application-scan",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
    "application",
    "web-browser",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": 0.01,
   "reward_info": {},
   "observation": 0
  },
  "ENG_HISTORIAN_USER": {
   "timestep": 28,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "ot_gateway",
    "application_name": "database-client"
   },
   "request": [
    "network",
    "node",
    "ot_gateway",
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
  "HMI_OPERATOR": {
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
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "BACKUP_VALIDATOR": {
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
   "reward": 0.3,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 28,
   "action": "node-service-fix",
   "parameters": {
    "node_name": "ot_controller",
    "service_name": "database-service"
   },
   "request": [
    "network",
    "node",
    "ot_controller",
    "service",
    "database-service",
    "fix"
   ],
   "response": {
    "status": "failure",
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 5,
        "source_wildcard_id": 3,
        "source_port_id": 1,
        "dest_ip_id": 2,
        "dest_wildcard_id": 2,
        "dest_port_id": 1,
        "protocol_id": 3
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
     }
    }
   }
  },
  "lateral_mover": {
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
  "dos_attacker": {
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
  "ransomware_attacker": {
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
  "campus_user": {
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
  "ENG_HISTORIAN_USER": {
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
  "HMI_OPERATOR": {
   "timestep": 29,
   "action": "node-application-scan",
   "parameters": {
    "node_name": "it_workstation",
    "application_name": "web-browser"
   },
   "request": [
    "network",
    "node",
    "it_workstation",
    "application",
    "web-browser",
    "scan"
   ],
   "response": {
    "status": "success",
    "data": {}
   },
   "reward": -0.3,
   "reward_info": {},
   "observation": 0
  },
  "BACKUP_VALIDATOR": {
   "timestep": 29,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "ot_controller",
    "application_name": "database-service"
   },
   "request": [
    "network",
    "node",
    "ot_controller",
    "application",
    "database-service",
    "execute"
   ],
   "response": {
    "status": "unreachable",
    "data": {
     "reason": [
      "Request ['database-service', 'execute'] could not be processed because database-service is not a valid request name",
      "within this RequestManager"
     ]
    }
   },
   "reward": 0.31,
   "reward_info": {},
   "observation": 0
  },
  "ot_monitor": {
   "timestep": 29,
   "action": "router-acl-add-rule",
   "parameters": {
    "target_router": "edge_router",
    "permission": "DENY",
    "protocol_name": "tcp",
    "src_ip": "10.0.20.1",
    "src_wildcard": "0.0.0.255",
    "src_port": "ALL",
    "dst_ip": "10.0.0.20",
    "dst_wildcard": "0.0.0.0",
    "dst_port": "ALL",
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
    "10.0.20.1",
    "0.0.0.255",
    "ALL",
    "10.0.0.20",
    "0.0.0.0",
    "ALL",
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
          "inbound": 0,
          "outbound": 0
         },
         "tcp": {
          "80": {
           "inbound": 0,
           "outbound": 0
          },
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 1,
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
         "inbound": 1,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
          "443": {
           "inbound": 0,
           "outbound": 0
          },
          "21": {
           "inbound": 0,
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
        "permission": 1,
        "source_ip_id": 1,
        "source_wildcard_id": 1,
        "source_port_id": 4,
        "dest_ip_id": 1,
        "dest_wildcard_id": 1,
        "dest_port_id": 4,
        "protocol_id": 1
       },
       "5": {
        "position": 5,
        "permission": 2,
        "source_ip_id": 5,
        "source_wildcard_id": 3,
        "source_port_id": 1,
        "dest_ip_id": 2,
        "dest_wildcard_id": 2,
        "dest_port_id": 1,
        "protocol_id": 3
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
     }
    }
   }
  },
  "lateral_mover": {
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
  "dos_attacker": {
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
  "ransomware_attacker": {
   "timestep": 29,
   "action": "node-application-execute",
   "parameters": {
    "node_name": "dmz_proxy",
    "application_name": "ransomware-script"
   },
   "request": [
    "network",
    "node",
    "dmz_proxy",
    "application",
    "ransomware-script",
    "execute"
   ],
   "response": {
    "status": "failure",
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