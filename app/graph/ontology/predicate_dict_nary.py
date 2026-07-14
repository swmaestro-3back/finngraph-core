PREDICATE_DICT_NARY: dict = {
    # 매각하다
    "DIVESTS_FROM": {
        "description": "Subject sells or divests its ownership stake in the object company.",
        "arguments": {
            "seller": {
                "types": [
                    "COMPANY",
                    "GOVERNMENT"
                ],
                "description": "Seller",
                "required": True
            },
            "company_being_sold": {
                "types": [
                    "COMPANY"
                ],
                "description": "Company being sold",
                "required": True
            }
        }
    },
    # 투자하다
    "INVESTS_IN": {
        "description": "Subject invests capital into the object.",
        "arguments": {
            "investor": {
                "types": [
                    "COMPANY",
                    "GOVERNMENT",
                    "COUNTRY"
                ],
                "description": "Investor",
                "required": True
            },
            "investment_recipient": {
                "types": [
                    "COMPANY",
                    "COUNTRY"
                ],
                "description": "Investment recipient",
                "required": True
            }
        }
    },
    # 분할하다
    "SPLITS_OFF": {
        "description": "Subject splits off the object as a separate company.",
        "arguments": {
            "splitting_company": {
                "types": [
                    "COMPANY"
                ],
                "description": "Splitting company",
                "required": True
            },
            "company_split_off": {
                "types": [
                    "COMPANY"
                ],
                "description": "Company split off",
                "required": True
            }
        }
    },
    # 협력하다
    "PARTNERS_WITH": {
        "description": "Subject cooperates with the object on a shared initiative, or signs a formal agreement/partnership with the object.",
        "arguments": {
            "cooperating_party": {
                "types": [
                    "COMPANY",
                    "GOVERNMENT",
                    "COUNTRY"
                ],
                "description": "Cooperating party",
                "required": True
            },
            "cooperation_partner": {
                "types": [
                    "COMPANY",
                    "GOVERNMENT",
                    "COUNTRY"
                ],
                "description": "Cooperation partner",
                "required": True
            }
        }
    },
    # 수주하다.
    "WINS_CONTRACT_FROM": {
        "description": "Subject wins a bid, tender, or order awarded by the object client. Extracted as a single event.",
        "arguments": {
            "winner": {
                "types": [
                    "COMPANY"
                ],
                "description": "Company that wins the contract",
                "required": True
            },
            "client": {
                "types": [
                    "COMPANY",
                    "GOVERNMENT",
                    "COUNTRY"
                ],
                "description": "Client or bid-awarding entity",
                "required": True
            },
            "item": {
                "types": [
                    "PRODUCT",
                    "COMMODITY"
                ],
                "description": "Ordered product or commodity",
                "required": False
            }
        }
    },
    # 인수하다
    "ACQUIRES": {
        "description": "Subject obtains ownership of the object company.",
        "arguments": {
            "acquirer": {
                "types": [
                    "COMPANY",
                    "GOVERNMENT"
                ],
                "description": "Acquirer",
                "required": True
            },
            "company_being_acquired": {
                "types": [
                    "COMPANY"
                ],
                "description": "Company being acquired",
                "required": True
            }
        }
    },
    # 공급하다
    "SUPPLIES_TO": {
        "description": "Supplier supplies an item to a recipient.",
        "arguments": {
            "supplier": {
                "types": [
                    "COMPANY"
                ],
                "description": "Supplier",
                "required": True
            },
            "recipient": {
                "types": [
                    "COMPANY",
                    "GOVERNMENT",
                    "COUNTRY"
                ],
                "description": "Recipient of the supply",
                "required": True
            },
            "item": {
                "types": [
                    "COMMODITY",
                    "PRODUCT"
                ],
                "description": "Product or material supplied",
                "required": True
            }
        }
    },
    # 수출하다
    "EXPORTS_TO": {
        "description": "Exporter exports an item to an importing party abroad. Extracted as a single event.",
        "arguments": {
            "exporter": {
                "types": [
                    "COMPANY",
                    "COUNTRY"
                ],
                "description": "Exporter",
                "required": True
            },
            "importing_party": {
                "types": [
                    "COMPANY",
                    "COUNTRY"
                ],
                "description": "Importing party",
                "required": True
            },
            "item": {
                "types": [
                    "COMMODITY",
                    "PRODUCT"
                ],
                "description": "Product or commodity exported",
                "required": True
            }
        }
    },
    # 수입하다
    "IMPORTS_FROM": {
        "description": "Importer imports an item from a supplying party abroad. Extracted as a single event.",
        "arguments": {
            "importer": {
                "types": [
                    "COMPANY",
                    "COUNTRY"
                ],
                "description": "Importer",
                "required": True
            },
            "supplying_party": {
                "types": [
                    "COMPANY"
                ],
                "description": "Supplying party",
                "required": True
            },
            "item": {
                "types": [
                    "COMMODITY",
                    "PRODUCT"
                ],
                "description": "Product or commodity imported",
                "required": True
            }
        }
    },
    # 위치하다
    "LOCATED_IN": {
        "description": "Subject is located or headquartered in the object country.",
        "arguments": {
            "located_entity": {
                "types": [
                    "COMPANY"
                ],
                "description": "Located entity",
                "required": True
            },
            "location": {
                "types": [
                    "COUNTRY"
                ],
                "description": "Location (country)",
                "required": True
            }
        }
    },
    # 생산하다
    "PRODUCES": {
        "description": "Subject manufactures or produces the object.",
        "arguments": {
            "producer": {
                "types": [
                    "COMPANY"
                ],
                "description": "Producer",
                "required": True
            },
            "produced_item": {
                "types": [
                    "PRODUCT",
                    "COMMODITY"
                ],
                "description": "Product or commodity produced",
                "required": True
            }
        }
    }
}
