import requests
from pathlib import Path


cookies = {
    "A1": "d=AQABBC0S6WQCEB9v3r3DduLQ_ADFJg8tBwgFEgEBAQFj6mTyZFkWyyMA_eMAAA&S=AQAAAi8K9SaoKvwUll9WAM3wvVo",
    "A3": "d=AQABBC0S6WQCEB9v3r3DduLQ_ADFJg8tBwgFEgEBAQFj6mTyZFkWyyMA_eMAAA&S=AQAAAi8K9SaoKvwUll9WAM3wvVo",
    "A1S": "d=AQABBC0S6WQCEB9v3r3DduLQ_ADFJg8tBwgFEgEBAQFj6mTyZFkWyyMA_eMAAA&S=AQAAAi8K9SaoKvwUll9WAM3wvVo&j=WORLD",
    "cmp": "t=1692996143&j=0&u=1---",
    "gpp": "DBAA",
    "gpp_sid": "-1",
    "thamba": "1",
    "PRF": "t%3DEICHERMOT.NS%26newChartbetateaser%3D0%252C1694205837297",
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/116.0",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://finance.yahoo.com/most-active?count=100&offset=100",
    "Content-Type": "application/json",
    "Origin": "https://finance.yahoo.com",
    "DNT": "1",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
}

params = {
    "crumb": "BG5F/eVe9qj",
    "lang": "en-US",
    "region": "US",
    "formatted": "true",
    "corsDomain": "finance.yahoo.com",
}
quotes = []

for x in range(0, 10000, 100):
    json_data = {
        "offset": x,
        "size": 100,
        "sortField": "dayvolume",
        "sortType": "DESC",
        "quoteType": "EQUITY",
        "query": {
            "operator": "AND",
            "operands": [
                {
                    "operator": "or",
                    "operands": [
                        {
                            "operator": "GT",
                            "operands": [
                                "intradaymarketcap",
                                100000000000,
                            ],
                        },
                    ],
                },
                {
                    "operator": "or",
                    "operands": [
                        {
                            "operator": "EQ",
                            "operands": [
                                "region",
                                "in",
                            ],
                        },
                    ],
                },
            ],
        },
        "userId": "",
        "userIdType": "guid",
    }

    response = requests.post(
        "https://query1.finance.yahoo.com/v1/finance/screener",
        params=params,
        cookies=cookies,
        headers=headers,
        json=json_data,
    ).json()

    quote_items = response.get("finance", {}).get("result", {})[0].get("quotes", {})
    if not quote_items:
        break

    quotes.extend([item["symbol"] for item in quote_items])
    Path("quotes").write_text("\n".join(quotes))
