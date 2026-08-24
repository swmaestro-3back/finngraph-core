# news

네이버 검색 API로 뉴스를 수집하고, 제목 기반으로 같은 사건을 다룬 기사끼리 묶는다.
LLM은 쓰지 않는다. Kiwi 형태소 분석 + TF-IDF + 코사인 유사도만 사용한다.

## 실행

```bash
uv run python -m news.pipeline --query "특징주,공급" --display 100
uv run python -m news.pipeline --from-file --threshold 0.30   # API 재호출 없이 재클러스터링
```

- `news/news.json` — 수집한 기사 원본
- `news/clusters.json` — 군집 결과(대표 기사, 키워드, 응집도 포함)

`.env`에 `NAVER_CLIENT_ID` / `NAVER_CLIENT_SECRET`이 있어야 한다.

## 파이프라인

| 모듈 | 역할 |
|---|---|
| `client.py` | 검색 API 호출. 100건 단위 페이징, link 기준 중복 제거 |
| `store.py` | news.json / clusters.json 입출력 |
| `preprocess.py` | HTML 제거 → Kiwi 명사 추출 → 복합명사 복원 → 불용어 제거 |
| `vectorize.py` | 가중 TF-IDF (numpy 직접 구현, L2 정규화) |
| `cluster.py` | average-link 병합 클러스터링 + 대표 기사(메도이드) 선정 |
| `pipeline.py` | 위 단계를 잇는 CLI |

## 설계 메모

**복합명사 복원.** Kiwi는 `LG에너지솔루션`을 `LG / 에너지 / 솔루션`으로 쪼갠다. 조각만
쓰면 서로 다른 기업이 겹쳐 보이므로, 원문에서 공백 없이 붙어 있던 명사 토큰들을 다시
이어 붙인 복합명사를 함께 만들고 가중치 3.0을 준다(고유명사 2.0, 일반명사 1.0).
특징주 기사는 제목에 종목명이 거의 반드시 들어가므로 이 신호가 사실상 군집을 결정한다.

**불용어.** `특징주 / 강세 / 급등 / 코스피` 같은 기사 형식어를 제거한다. IDF로도 상당
부분 눌리지만, 남겨두면 복합명사 결합이 오염된다(`특징주삼성전자`).

**average-link.** single-link의 사슬 효과(A~B, B~C가 가까우면 A~C가 멀어도 한 덩어리)를
피한다. threshold를 잘라내는 방식이라 군집 수를 미리 정할 필요가 없다.

**본문.** 검색 API는 본문을 주지 않는다. `description`(200자 안팎 요약)을 가중치 0.4로
보조 신호로 쓴다. 전문이 필요하면 `NewsItem.body` 필드를 비워뒀으니, `link`
(n.news.naver.com)의 `#dic_area`를 파싱하는 크롤러를 붙여 채우면 된다.

## 튜닝

`--threshold` 기본값 0.35. 100건 기준 대략:

| threshold | 군집 수 | 단독 기사 |
|---|---|---|
| 0.25 | 58 | 45 |
| 0.30 | 60 | 48 |
| 0.35 | 62 | 51 |
| 0.40 | 65 | 54 |

낮출수록 같은 종목의 다른 사건까지 한 덩어리가 되고, 높일수록 같은 사건이 쪼개진다.
`--description-weight 0`을 주면 순수 제목만으로 묶는다.
