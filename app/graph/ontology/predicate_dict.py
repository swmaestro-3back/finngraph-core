PREDICATE_DICT: dict = {
    "인수하다": {
        "subject": [
            "COMPANY",
            "GOVERNMENT"
        ],
        "object": [
            "COMPANY"
        ],
        "description": "Subject acquires ownership or control of the object company.",
        "subject_role": "Acquirer",
        "object_role": "Company being acquired"
    },
    "설립하다": {
        "subject": [
            "COMPANY",
            "GOVERNMENT"
        ],
        "object": [
            "COMPANY"
        ],
        "description": "Subject founds or establishes the object company.",
        "subject_role": "Founder",
        "object_role": "Company being founded"
    },
    "합병하다": {
        "subject": [
            "COMPANY"
        ],
        "object": [
            "COMPANY"
        ],
        "description": "Subject merges with the object company into one entity.",
        "subject_role": "Merging company",
        "object_role": "Company being merged with"
    },
    "매각하다": {
        "subject": [
            "COMPANY",
            "GOVERNMENT"
        ],
        "object": [
            "COMPANY"
        ],
        "description": "Subject sells or divests its ownership stake in the object company.",
        "subject_role": "Seller",
        "object_role": "Company being sold"
    },
    "투자하다": {
        "subject": [
            "COMPANY",
            "GOVERNMENT",
            "COUNTRY"
        ],
        "object": [
            "COMPANY",
            "COUNTRY"
        ],
        "description": "Subject invests capital into the object.",
        "subject_role": "Investor",
        "object_role": "Investment recipient"
    },
    "계약하다": {
        "subject": [
            "COMPANY",
            "GOVERNMENT"
        ],
        "object": [
            "COMPANY",
            "GOVERNMENT"
        ],
        "description": "Subject enters into a contract or agreement with the object.",
        "subject_role": "Contracting party",
        "object_role": "Counterparty"
    },
    "분할하다": {
        "subject": [
            "COMPANY"
        ],
        "object": [
            "COMPANY"
        ],
        "description": "Subject splits off the object as a separate company.",
        "subject_role": "Splitting company",
        "object_role": "Company split off"
    },
    "제휴하다": {
        "subject": [
            "COMPANY",
            "GOVERNMENT"
        ],
        "object": [
            "COMPANY",
            "GOVERNMENT"
        ],
        "description": "Subject forms a business alliance or partnership with the object.",
        "subject_role": "Allying party",
        "object_role": "Alliance partner"
    },
    "수주하다": {
        "subject": [
            "COMPANY"
        ],
        "object": [
            "PRODUCT",
            "COMMODITY"
        ],
        "description": "Subject wins an order or contract to supply the object.",
        "subject_role": "Contract recipient",
        "object_role": "Ordered product or commodity"
    },
    "취득하다": {
        "subject": [
            "COMPANY",
            "GOVERNMENT"
        ],
        "object": [
            "COMPANY"
        ],
        "description": "Subject obtains ownership of the object company (near-synonym of 인수하다).",
        "subject_role": "Acquirer",
        "object_role": "Company being acquired"
    },
    "매입하다": {
        "subject": [
            "COMPANY",
            "GOVERNMENT"
        ],
        "object": [
            "COMPANY"
        ],
        "description": "Subject purchases the object company or a stake in it.",
        "subject_role": "Buyer",
        "object_role": "Company being bought"
    },
    "분사하다": {
        "subject": [
            "COMPANY"
        ],
        "object": [
            "COMPANY"
        ],
        "description": "Subject spins off the object as an independent company.",
        "subject_role": "Parent company",
        "object_role": "Spun-off company"
    },
    "협력하다": {
        "subject": [
            "COMPANY",
            "GOVERNMENT",
            "COUNTRY"
        ],
        "object": [
            "COMPANY",
            "GOVERNMENT",
            "COUNTRY"
        ],
        "description": "Subject cooperates with the object on a shared initiative.",
        "subject_role": "Cooperating party",
        "object_role": "Cooperation partner"
    },
    "유치하다": {
        "subject": [
            "COMPANY",
            "GOVERNMENT",
            "COUNTRY"
        ],
        "object": [
            "COMPANY"
        ],
        "description": "Subject attracts or recruits the object (e.g., investment, a company, or a facility).",
        "subject_role": "Recruiting party",
        "object_role": "Entity being attracted"
    },
    "체결하다": {
        "subject": [
            "COMPANY",
            "GOVERNMENT"
        ],
        "object": [
            "COMPANY",
            "GOVERNMENT"
        ],
        "description": "Subject signs or concludes a formal agreement with the object.",
        "subject_role": "Signing party",
        "object_role": "Counterparty"
    },
    "낙찰받다": {
        "subject": [
            "COMPANY"
        ],
        "object": [
            "COMPANY",
            "GOVERNMENT"
        ],
        "description": "Subject wins a bid or tender awarded by the object.",
        "subject_role": "Bid winner",
        "object_role": "Bid-awarding entity"
    },
    "입찰하다": {
        "subject": [
            "COMPANY"
        ],
        "object": [
            "COMPANY",
            "GOVERNMENT"
        ],
        "description": "Subject places a bid in a tender held by the object.",
        "subject_role": "Bidder",
        "object_role": "Tender-holding entity"
    },
    "공급하다": {
        "subject": [
            "COMPANY"
        ],
        "object": [
            "COMPANY",
            "GOVERNMENT",
            "COUNTRY",
            "COMMODITY",
            "PRODUCT"
        ],
        "description": "Subject supplies the object, whether a recipient buyer or the product/material supplied.",
        "subject_role": "Supplier",
        "object_role": "Recipient, or the product/material supplied"
    },
    "납품하다": {
        "subject": [
            "COMPANY"
        ],
        "object": [
            "COMPANY",
            "GOVERNMENT",
            "COUNTRY",
            "COMMODITY",
            "PRODUCT"
        ],
        "description": "Subject delivers goods to the object as a vendor, whether the client or the delivered product/material.",
        "subject_role": "Vendor",
        "object_role": "Client, or the delivered product/material"
    },
    "제공하다": {
        "subject": [
            "COMPANY",
            "GOVERNMENT"
        ],
        "object": [
            "COMPANY",
            "GOVERNMENT",
            "COMMODITY",
            "PRODUCT"
        ],
        "description": "Subject provides a service, product, or resource to the object.",
        "subject_role": "Provider",
        "object_role": "Recipient, or the product/service provided"
    },
    "판매하다": {
        "subject": [
            "COMPANY"
        ],
        "object": [
            "COMPANY",
            "GOVERNMENT",
            "COUNTRY",
            "COMMODITY",
            "PRODUCT"
        ],
        "description": "Subject sells the object, whether a buyer or the product/commodity sold.",
        "subject_role": "Seller",
        "object_role": "Buyer, or the product/commodity sold"
    },
    "구매하다": {
        "subject": [
            "COMPANY",
            "GOVERNMENT",
            "COUNTRY"
        ],
        "object": [
            "COMPANY",
            "COMMODITY",
            "PRODUCT"
        ],
        "description": "Subject purchases the object, whether a company or a product/commodity.",
        "subject_role": "Buyer",
        "object_role": "Company acquired, or the product/commodity purchased"
    },
    "조달하다": {
        "subject": [
            "COMPANY",
            "GOVERNMENT"
        ],
        "object": [
            "COMPANY",
            "COMMODITY",
            "PRODUCT"
        ],
        "description": "Subject procures or sources the object, whether a company or materials.",
        "subject_role": "Procuring party",
        "object_role": "Company or material procured"
    },
    "위탁하다": {
        "subject": [
            "COMPANY",
            "GOVERNMENT"
        ],
        "object": [
            "COMPANY",
            "COMMODITY",
            "PRODUCT"
        ],
        "description": "Subject outsources or entrusts production/handling of the object to another party.",
        "subject_role": "Outsourcing party",
        "object_role": "Entrusted company or product"
    },
    "유통하다": {
        "subject": [
            "COMPANY"
        ],
        "object": [
            "COMPANY",
            "COMMODITY",
            "PRODUCT"
        ],
        "description": "Subject distributes the object in the market, whether a partner company or a product/commodity.",
        "subject_role": "Distributor",
        "object_role": "Distribution partner, or the product/commodity distributed"
    },
    "의존하다": {
        "subject": [
            "COMPANY",
            "GOVERNMENT",
            "COUNTRY"
        ],
        "object": [
            "COMPANY",
            "COUNTRY",
            "COMMODITY",
            "PRODUCT"
        ],
        "description": "Subject depends or relies on the object for supply or support.",
        "subject_role": "Dependent party",
        "object_role": "Party or resource depended on"
    },
    "수출하다": {
        "subject": [
            "COMPANY"
        ],
        "object": [
            "COMPANY",
            "COUNTRY",
            "COMMODITY",
            "PRODUCT"
        ],
        "description": "Subject exports the object abroad, whether a buyer/country or a product/commodity.",
        "subject_role": "Exporter",
        "object_role": "Importing party, or the product/commodity exported"
    },
    "수입하다": {
        "subject": [
            "COMPANY",
            "COUNTRY"
        ],
        "object": [
            "COMPANY",
            "COMMODITY",
            "PRODUCT"
        ],
        "description": "Subject imports the object from abroad, whether a supplier or a product/commodity.",
        "subject_role": "Importer",
        "object_role": "Supplying party, or the product/commodity imported"
    },
    "위치하다": {
        "subject": [
            "COMPANY"
        ],
        "object": [
            "COUNTRY"
        ],
        "description": "Subject is located or headquartered in the object country.",
        "subject_role": "Located entity",
        "object_role": "Location (country)"
    },
    "생산하다": {
        "subject": [
            "COMPANY"
        ],
        "object": [
            "PRODUCT",
            "COMMODITY"
        ],
        "description": "Subject manufactures or produces the object.",
        "subject_role": "Producer",
        "object_role": "Product or commodity produced"
    }
}
