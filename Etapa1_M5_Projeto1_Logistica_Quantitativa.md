# Projeto 1 — Logística Quantitativa Aplicada (Etapa 1)
## Foco: **M5 Forecasting (Walmart)** — Previsão de Demanda + Estoques Estocásticos + Pooling + Localização (proxy) + Desenho Computacional (C3)

**Disciplina:** Logística Quantitativa Aplicada — UnB / FT / EPR  
**Professor:** João Gabriel de Moraes Souza  
**Discente:** Antônio Augusto Maciel Guimarães 190084421  
**Data:** 31/01/2026  
**Entrega:** Etapa 1 — Formulação, Modelagem e Desenho Computacional

---

## Sumário
1. [Formulação do problema logístico](#1-formulação-do-problema-logístico)  
2. [Justificativa do tópico central](#2-justificativa-do-tópico-central)  
3. [Base de dados (M5) e estratégia de recorte](#3-base-de-dados-m5-e-estratégia-de-recorte)  
4. [Análise exploratória (EDA) — plano e entregáveis](#4-análise-exploratória-eda--plano-e-entregáveis)  
5. [Previsão de demanda (quando pertinente)](#5-previsão-de-demanda-quando-pertinente)  
6. [Modelagem determinística (baseline): EOQ + custo total](#6-modelagem-determinística-baseline-eoq--custo-total)  
7. [Discussão preliminar: incerteza e risco](#7-discussão-preliminar-incerteza-e-risco)  
8. [Desenho computacional (C3): arquitetura, módulos e pseudocódigo](#8-desenho-computacional-c3-arquitetura-módulos-e-pseudocódigo)  
9. [Como a incerteza entrará na Etapa 2 (simulação/avaliação)](#9-como-a-incerteza-entrará-na-etapa-2-simulaçãoavaliação)  
10. [Checklist de conformidade com a Etapa 1](#10-checklist-de-conformidade-com-a-etapa-1)  
11. [Referências e links](#11-referências-e-links)
12. [APÊNDICE C — Implementação Completa (Código Executável)](#apêndice-c--implementação-completa-código-executável)

---

## Resumo executivo
Este projeto trata do problema de **reposicionamento de estoque** em uma rede varejista com **alta variabilidade de demanda**, influenciada por **eventos de calendário** e **mudança de preços**, com estrutura **hierárquica** (produto e geografia). Utiliza-se o dataset **M5 Forecasting (Walmart)** (Kaggle), que fornece vendas diárias por item e loja, além de calendário e preços.

O objetivo é propor uma política de reposição que **minimize custo total** e atinja um **nível de serviço** alvo. Para isso, serão integrados:
- **Previsão de demanda (II):** modelos de séries temporais e regressoras (preço/calendário), com avaliação por janela temporal;
- **Modelo determinístico (III):** EOQ como baseline e referência de custo;
- **Modelo estocástico (IV):** política (R,Q) com estoque de segurança derivado de variância (resíduos do forecast);
- **Centralização / risk pooling (VII):** comparar estoque descentralizado vs pooled (por estado/CD);
- **Localização (VI):** cenário com CDs candidatos e custo de transporte por proxy (Etapa 1), evoluindo para distâncias reais (Etapa 2);
- **Simulação (V):** planejada para a Etapa 2 via SimPy, incorporando incerteza de demanda e lead time.

---

## 1. Formulação do problema logístico
### 1.1 Contexto operacional
Uma rede varejista opera múltiplas lojas e milhares de produtos com vendas diárias. A demanda é influenciada por:
- sazonalidade semanal e anual;
- eventos e feriados;
- promoções e alterações de preço;
- efeitos regionais (estado/loja).

Decisões de reposição afetam diretamente:
- **rupturas** (perda de venda e degradação de serviço);
- **excesso de estoque** (custo de capital + armazenagem + risco de obsolescência);
- custo de pedidos e logística (CD/transportes).

### 1.2 Objetivo geral
Definir uma política de reposição que minimize custo logístico e garanta nível de serviço, integrando previsão, estoque e cenários de centralização/localização.

### 1.3 Função objetivo (custo total esperado)
No horizonte \(T\), minimizar:
\[
\min \ \mathbb{E}[C_{total}] = \mathbb{E}[C_{pedido} + C_{holding} + C_{ruptura} + C_{transporte}]
\]

Onde:
- \(C_{pedido}\): custo fixo por pedido (setup/ordem);
- \(C_{holding}\): custo de manter estoque (h por unidade por período);
- \(C_{ruptura}\): penalidade por falta (custo de oportunidade ou backorder);
- \(C_{transporte}\): custo variável (proxy por distância × volume).

### 1.4 Variáveis de decisão
Por item \(i\) e loja \(s\):
- \(Q_{i,s}\): lote de reposição (quantidade por pedido)
- \(R_{i,s}\): ponto de pedido (reorder point)

Em cenários de centralização/localização:
- \(x_{s,c}\in\{0,1\}\): loja \(s\) atendida pelo CD \(c\)
- (opcional) \(y_c\in\{0,1\}\): abertura do CD \(c\) em local candidato

### 1.5 Restrições e nível de serviço
**Nível de serviço (meta):**
- **CSL**: \(P(\text{não faltar no ciclo}) \ge \alpha\) (ex.: 95%)
ou
- **Fill rate \(\beta\)**: fração atendida sem falta \(\ge \beta\) (ex.: 98%)

**Restrições operacionais (selecionar para o recorte):**
- capacidade de estoque por loja \(I_{s}^{max}\);
- orçamento por ciclo \(\sum_i c_i Q_{i,s}\le B_s\);
- lote mínimo/múltiplo (embalagem, paletização).

### 1.6 Hipóteses
- Lead time \(L\) **determinístico** na Etapa 1; \(L\) **estocástico** na Etapa 2.
- Custos \(K, h, p\) definidos por cenário e analisados via sensibilidade.
- Em recortes, assume-se independência parcial entre lojas; correlação será medida e discutida.

---

## 2. Justificativa do tópico central
### Tópico central: **IV — Gestão de Estoques Estocásticos**
O eixo do projeto é o tópico IV por refletir melhor o problema real: a demanda (e o lead time) são incertos, e a política ótima depende do trade-off custo × serviço × risco.

**Interfaces:**
- **I (Fundamentos):** trade-offs, métricas (CSL/fill rate), custo total e estrutura do sistema logístico;
- **II (Previsão de demanda):** estima \(\mu\) e \(\sigma\) por série/grupo;
- **III (Determinístico):** EOQ e custo total como baseline e comparador;
- **V (Simulação):** validação de políticas e níveis de serviço sob incerteza (Etapa 2);
- **VI (Localização):** desenho de CDs candidatos e alocação de lojas (custo de transporte + efeito no lead time);
- **VII (Centralização / Risk pooling):** redução de variância efetiva e estoque de segurança total.

---

## 3. Base de dados (M5) e estratégia de recorte
### 3.1 Link e acesso
- Dataset M5 (Kaggle): `https://www.kaggle.com/competitions/m5-forecasting-accuracy/data`

### 3.2 Arquivos principais (M5 Accuracy)
- `sales_train_validation.csv` (vendas diárias históricas — formato wide)
- `calendar.csv` (data → eventos/feriados/features)
- `sell_prices.csv` (preço por item/store/semana)
- `sample_submission.csv` (formato de previsão)

### 3.3 Estrutura hierárquica (complexidade)
O M5 permite análises em múltiplos níveis, por exemplo:
- **Geografia:** store → state → total
- **Catálogo:** item → dept → cat → total

### 3.4 Recorte proposto
Para manter robustez e viabilidade computacional:
- **Recorte A (principal):** 2 lojas no mesmo estado + 2 categorias + ~30 itens com maior volume;
- **Recorte B (intermitência):** 10 itens com alta % de zeros;
- **Recorte C (pooling):** mesmas categorias comparando *descentralizado* vs *pooling por estado/CD*.

> A lógica do recorte é: (i) preservar hierarquia e regressoras, (ii) permitir comparação de políticas, (iii) manter o pipeline reprodutível para Etapa 2.

---

## 4. Análise exploratória (EDA) — plano e entregáveis
### 4.1 Pré-processamento (requisito para EDA)
1. Converter `sales_train_validation` de wide → long:
   - colunas: `date`, `id`, `store_id`, `state_id`, `cat_id`, `dept_id`, `item_id`, `demand`
2. Fazer merge com `calendar.csv` (features temporais/eventos).
3. Fazer merge com `sell_prices.csv` por (`store_id`, `item_id`, `wm_yr_wk`).
4. Criar features derivadas:
   - `dow`, `month`, `week_of_year`, `is_weekend`, `is_event`, `snap_state` etc.
   - `price_change`, `price_index`, `promo_proxy` (se aplicável).

### 4.2 Diagnósticos por série (store×item)
Entregáveis (figuras/tabelas):
- Série temporal (demanda diária) e agregações semanais;
- Decomposição sazonal (semanal/anual);
- Estatísticas: média, desvio, coeficiente de variação (CV);
- **Intermitência:** % de zeros, tamanho médio de "runs" sem demanda, intervalo médio entre vendas;
- Outliers (picos) e associação com eventos/preço.

### 4.3 Diagnósticos por nível hierárquico
Entregáveis:
- Demanda agregada por `cat_id` e `dept_id` ao longo do tempo;
- Comparação entre lojas (diferença de padrão e correlação);
- Relação preço × demanda (elasticidade aproximada por grupo).

### 4.4 Medidas quantitativas (para "nota alta")
- **ACF/PACF** em séries selecionadas;
- **Correlação cruzada** entre lojas para mesmos itens;
- **Teste de mudança de regime** (breakpoints) em séries de alto giro;
- **Mapa de calor** de sazonalidade (dow×mês) por categoria.

---

## 5. Previsão de demanda (quando pertinente)
### 5.1 Objetivo da previsão
Prever demanda futura \(\hat{D}_{i,s}(t)\) e obter medida de incerteza (variância) para suportar estoque de segurança.

### 5.2 Estratégia de modelagem (escada de complexidade)
**(A) Baselines obrigatórios (comparação justa):**
- Naive e Seasonal Naive (sazonalidade semanal);
- Média móvel;
- Suavização exponencial (ETS).

**(B) Modelos por série (locais):**
- ARIMA/SARIMA (onde houver padrão);
- ARIMAX/Regressão com regressoras (preço + eventos + dow).

**(C) Modelos globais/híbridos (ponto avançado):**
- modelo "global" com pooling estatístico entre séries (por categoria/loja);
- abordagem em painel com efeitos fixos por loja/item;
- (opcional) LightGBM com features de calendário/preço e lags.

### 5.3 Validação e métricas
- Split temporal (treino → validação) com janela rolante;
- Métricas recomendadas:
  - MAE / RMSE
  - WAPE (robusto para escalas)
  - SMAPE (cuidado com zeros; reportar junto com WAPE)
- Diagnóstico de resíduos:
  - autocorrelação remanescente;
  - heterocedasticidade por dia da semana/evento;
  - distribuição dos erros (caudas → risco de ruptura).

### 5.4 Saída crítica para estoques: \(\mu\) e \(\sigma\)
A política de estoque utilizará:
- \(\mu\): média prevista da demanda;
- \(\sigma\): variância estimada via resíduos (ou bootstrap/empírico).

---

## 6. Modelagem determinística (baseline): EOQ + custo total
Mesmo com foco estocástico, um baseline determinístico é necessário para comparação e argumento técnico.

### 6.1 EOQ por item-loja
\[
Q^*_{i,s}=\sqrt{\frac{2K_{i,s}D_{i,s}}{h_{i,s}}}
\]
- \(D_{i,s}\): demanda anual estimada (\(\bar{d}_{i,s}\cdot 365\))
- \(K_{i,s}\): custo por pedido (setup)
- \(h_{i,s}\): holding anual por unidade

### 6.2 Custo total determinístico
\[
CT(Q)=\frac{K D}{Q}+\frac{hQ}{2}
\]
Interpretação:
- \(KD/Q\): custo de pedidos (quanto menor Q, mais pedidos)
- \(hQ/2\): custo médio de holding

### 6.3 Como o determinístico vira "ponte" para o estocástico
- EOQ define \(Q\) como lote economicamente eficiente **sem incerteza**
- Em seguida, introduz-se o risco via \(R\) e estoque de segurança (Seção 7)

---

## 7. Discussão preliminar: incerteza e risco
### 7.1 Principais fontes de incerteza
- **Demanda:** sazonalidade, eventos, intermitência, promoções, mudança de preço;
- **Lead time:** variação logística, atrasos e rupturas do fornecedor;
- **Correlação entre lojas:** afeta o ganho real de pooling (risk pooling);
- **Erros de previsão:** caudas pesadas e picos podem dominar o risco.

### 7.2 Política estocástica (R,Q) — estoque de segurança
Assumindo demanda durante lead time aproximadamente Normal (baseline):
\[
\mu_L=\mu\cdot L,\quad \sigma_L=\sigma\cdot \sqrt{L}
\]
\[
R = \mu_L + z_{\alpha}\cdot \sigma_L
\]
Onde:
- \(\alpha\) é o nível CSL (ex.: 95%)
- \(z_{\alpha}\) é o quantil Normal correspondente
- \(SS=z_{\alpha}\sigma_L\)

### 7.3 Intermitência (diferencial do M5)
Como muitas séries têm % zeros alta, o projeto prevê:
- diagnóstico de intermitência;
- uso de distribuição empírica (amostragem) na Etapa 2;
- alternativa: Croston/TSB/Tweedie (conforme recorte e resultados).

### 7.4 Centralização / risk pooling (trade-off custo × serviço)
**Cenários comparados:**
1. Descentralizado: SS por loja;
2. Pooling por estado/CD: SS em agregado + distribuição;
3. Híbrido: itens A (alto giro) descentralizados; B/C centralizados.

Métrica-chave:
- comparar \(SS_{sum}=\sum_s SS_s\) vs \(SS_{pooled}\)
- medir correlação entre lojas para explicar quando pooling reduz pouco.

### 7.5 Matriz de risco (qualitativa + quantitativa)
Sugestão de matriz para o relatório:
- eixo X: probabilidade (baixa→alta) — estimada por frequência (eventos, outliers)
- eixo Y: impacto (custo de ruptura, dias sem estoque, perda de venda)
- mitigação: aumentar CSL, pooling, revisão de lead time, políticas (s,S).

---

## 8. Desenho computacional (C3): arquitetura, módulos e pseudocódigo
### 8.1 Arquitetura (pipeline reprodutível)
**Entrada:** dados M5 (vendas + calendário + preços)  
**Saídas Etapa 1:** relatório + plano C3 + especificação de políticas e cenários  
**Saídas Etapa 2:** simulação (SimPy), métricas e recomendação final

### 8.2 Estrutura de pastas (IMPLEMENTADA - Etapas 1 e 2)
```text
projeto_m5/
  data/
    raw/                    # Dados originais M5 (CSV)
    processed/              # Dados processados (pickle)
  src/
    config.py               # Parâmetros do projeto
    io_load.py              # Carregamento dos dados
    preprocess.py           # Pré-processamento e feature engineering
    eda/
      eda_core.py           # Estatísticas e CV
      plots.py              # Visualizações
      intermittency.py      # Análise de intermitência (ADI, CV²)
    forecast/
      baselines.py          # Naive, Seasonal Naive, Média Móvel
      ets_arima.py          # ETS e ARIMA/SARIMA
      exogenous.py          # ARIMAX com regressoras
      metrics.py            # MAE, RMSE, WAPE, SMAPE
      advanced.py           # LightGBM, Ensemble, Decomposição Sazonal
    inventory/
      eoq.py                # EOQ e custo determinístico
      rq_policy.py          # Política (R,Q) estocástica
      service_levels.py     # CSL e Fill Rate
      pooling.py            # Risk Pooling básico
      pooling_advanced.py   # [NOVO] Correlação, ABC híbrido, sensibilidade
      optimization.py       # [NOVO] Q* com custo de falta, multi-item
    location/
      candidates.py         # CDs candidatos e coordenadas
      allocation.py         # Alocação loja→CD
      transport_cost_proxy.py # Custo de transporte
    simulation/              # [NOVO - Etapa 2]
      __init__.py           # Módulo de exportação
      simpy_env.py          # Ambiente SimPy, políticas, Monte Carlo
  notebooks/
    01_preprocess.ipynb     # Notebook de pré-processamento
    02_eda.ipynb            # Notebook de EDA
    03_forecast.ipynb       # Notebook de previsão
    04_inventory.ipynb      # Notebook de estoque
  results/                  # Resultados e figuras
  main.py                   # Pipeline principal
  dashboard.py              # Dashboard interativo (5 páginas)
  requirements.txt          # Dependências (14 pacotes)
  README.md
```

### 8.3 Pseudocódigo (fluxo principal)
```text
MAIN():
  cfg = load_config()

  # 1) Carregar e preprocessar
  sales, calendar, prices = load_m5_raw(cfg)
  df = to_long_format(sales)
  df = merge_calendar_prices(df, calendar, prices)
  df = add_features(df)

  # 2) EDA + seleção de recortes
  eda_report(df)
  recortes = select_slices(df, strategy="A/B/C")

  # 3) Forecast
  for slice in recortes:
    train, val = time_split(slice)
    model = fit_forecast(train, models=["seasonal_naive","ETS","ARIMAX"])
    yhat = predict(model, horizon=H)
    resid = compute_residuals(model, val)
    store_sigma(slice.id, resid)

  # 4) Estoque determinístico (EOQ)
  for series in recortes:
    D = annualize_mean(series)
    Q = eoq(D, K, h)

  # 5) Estoque estocástico (R,Q)
  for series in recortes:
    mu = forecast_mean(series)
    sigma = forecast_sigma(series)   # resíduos / empírico
    R = reorder_point(mu, sigma, L, CSL)

  # 6) Pooling (cenários)
  compare_decentral_vs_pooled(recortes, grouping=["state","store","cat"])

  # 7) Localização (proxy Etapa 1)
  evaluate_location_scenarios(recortes, cd_candidates, transport_cost_proxy)

  export_stage1_outputs()
```

### 8.4 Bibliotecas previstas (INSTALADAS)
- `pandas`, `numpy`: manipulação e estatística;
- `statsmodels`: ETS/ARIMA e diagnósticos;
- `scikit-learn`: validação e modelos globais;
- `lightgbm`: modelo de ensemble para previsão avançada;
- `streamlit`, `plotly`: dashboard interativo;
- `simpy` (Etapa 2): simulação de eventos discretos;
- `pulp`/`ortools` (opcional): alocação/localização;
- `matplotlib`, `seaborn`: gráficos e relatórios.

---

## 9. Como a incerteza entrará na Etapa 2 (simulação/avaliação)
### 9.1 Modelagem estocástica na simulação
- **Demanda:** distribuição empírica por série (amostragem) + cenários (eventos/picos);
- **Lead time:** distribuição (ex.: triangular ou normal truncada) por cenário;
- **Políticas:** (R,Q), (s,S) e (P,S) como alternativas comparadas.

### 9.2 Experimentos e métricas (saídas da simulação)
Métricas de desempenho:
- custo total (pedido + holding + ruptura + transporte);
- CSL e fill rate;
- dias com falta e backorders;
- estoque médio, máximo e variância;
- sensibilidade por CSL (90/95/98), por lead time e por pooling.

Monte Carlo:
- N repetições por cenário;
- intervalos de confiança das métricas para decisão robusta.

---

## 10. Checklist de conformidade com a Etapa 1
- [x] **Formulação do problema** (objetivos, hipóteses e restrições)  
- [x] **Justificativa do tópico central** (IV) + interfaces com outros tópicos  
- [x] **Descrição e EDA** do dataset (M5) + recortes e entregáveis  
- [x] **Previsão de demanda** (modelos + métricas + resíduos → \(\sigma\))  
- [x] **Modelo determinístico** (EOQ + custo total)  
- [x] **Discussão de risco/incerteza** (demanda, lead time, erro, pooling)  
- [x] **Desenho computacional (C3)** (arquitetura, módulos, pseudocódigo, libs)  
- [x] **Plano explícito de como incorporar incerteza na Etapa 2** (SimPy + Monte Carlo)
- [x] **Implementação funcional** (código Python executável)
- [x] **Dashboard interativo** (Streamlit para visualização)

---

## 11. Referências e links
- Kaggle — M5 Forecasting (Data): `https://www.kaggle.com/competitions/m5-forecasting-accuracy/data`
- Statsmodels (séries temporais): `https://www.statsmodels.org`
- SimPy (simulação de eventos discretos): `https://simpy.readthedocs.io`
- OpenStreetMap (dados geográficos): `https://www.openstreetmap.org`
- LightGBM (gradient boosting): `https://lightgbm.readthedocs.io`
- Streamlit (dashboard): `https://streamlit.io`

---

## Apêndice A — Parâmetros econômicos (como definir sem "inventar")
Como o dataset não traz custos diretamente, serão utilizados **cenários parametrizados** e análise de sensibilidade:
- \(K\): custo fixo por pedido (ex.: 50, 100, 200) unidades monetárias
- \(h\): holding anual (ex.: 20% do valor unitário; ou 0.5–2.0 por unidade/ano)
- \(p\): penalidade de falta (ex.: margem perdida; 2× a 10× do holding diário)
- \(L\): lead time (ex.: 3, 7, 14 dias) e depois distribuição (Etapa 2)

A validade do projeto vem de:
- coerência metodológica;
- transparência dos cenários;
- robustez dos resultados por sensibilidade.

---

## Apêndice B — Sugestão de figuras (para o relatório final)
1. Série temporal (demanda) de 3 itens (alto giro, médio, intermitente)  
2. Heatmap sazonalidade (dow×mês) por categoria  
3. Dispersão preço×demanda e elasticidade aproximada  
4. Curva custo total determinístico (EOQ) vs custo com SS (estocástico)  
5. Comparação de SS total: descentralizado vs pooling  
6. Diagrama do pipeline (dados → forecast → EOQ → (R,Q) → cenários → simulação)

---

## Apêndice C — Implementação Completa (Código Executável)

### C.1 Visão Geral da Implementação
O projeto foi completamente implementado em Python, com código modular e documentado em português brasileiro. A implementação inclui:

| Módulo | Arquivo | Descrição |
|--------|---------|-----------|
| **Configuração** | `src/config.py` | Paths, lojas/categorias selecionadas, parâmetros econômicos (K, h, p) |
| **I/O** | `src/io_load.py` | Carregamento dos 3 arquivos CSV do M5 |
| **Pré-processamento** | `src/preprocess.py` | Filtros, melt (wide→long), merge, feature engineering |
| **EDA** | `src/eda/eda_core.py` | Estatísticas: média, desvio, CV, correlação |
| **EDA** | `src/eda/plots.py` | Séries temporais, heatmaps de sazonalidade |
| **EDA** | `src/eda/intermittency.py` | ADI, CV², classificação (Suave/Errático/Grumoso/Intermitente) |
| **Previsão** | `src/forecast/baselines.py` | Naive, Seasonal Naive, Média Móvel |
| **Previsão** | `src/forecast/ets_arima.py` | ETS (ExponentialSmoothing), ARIMA/SARIMA |
| **Previsão** | `src/forecast/exogenous.py` | ARIMAX com variáveis exógenas |
| **Previsão** | `src/forecast/advanced.py` | LightGBM, Ensemble, Decomposição Sazonal |
| **Previsão** | `src/forecast/metrics.py` | MAE, RMSE, WAPE, SMAPE |
| **Estoque** | `src/inventory/eoq.py` | EOQ e custo total determinístico |
| **Estoque** | `src/inventory/rq_policy.py` | Política (R,Q): SS, ponto de ressuprimento |
| **Estoque** | `src/inventory/service_levels.py` | CSL e Fill Rate |
| **Estoque** | `src/inventory/pooling.py` | Variância pooled e portfolio effect |
| **Localização** | `src/location/candidates.py` | Coordenadas de lojas e CDs, Haversine |
| **Localização** | `src/location/allocation.py` | Atribuição loja→CD mais próximo |
| **Localização** | `src/location/transport_cost_proxy.py` | Custo de transporte (fixo + variável) |

### C.2 Dashboard Interativo (Streamlit)
Um dashboard moderno foi desenvolvido com as seguintes funcionalidades:

**Páginas disponíveis:**
1. **🏠 Visão Geral** — Métricas globais, demanda por loja, intermitência, série temporal agregada
2. **📈 Análise de Demanda** — Série por item/loja, estatísticas, sazonalidade semanal/mensal
3. **🔮 Previsão** — 4 modelos (ETS, LightGBM, Ensemble, Decomposição), métricas, comparação
4. **📦 Gestão de Estoques** — EOQ, política (R,Q), análise de sensibilidade

**Parâmetros configuráveis (sidebar):**

| Parâmetro | Tipo | Range | Onde é Usado |
|-----------|------|-------|--------------|
| **Lead Time** | slider | 1–14 dias | Todas as páginas: demanda no LT, σ_L, SS, previsão |
| **CSL** | slider | 80%–99% | z-score para SS, análise de sensibilidade |
| **Horizonte** | select | 7/14/21/28 dias | Horizonte padrão de previsão |
| **K (custo pedido)** | input | R$10–500 | EOQ, custo total |
| **h (holding %)** | slider | 10%–50% a.a. | EOQ, custo total |

**Execução:**
```bash
conda activate logistica
streamlit run dashboard.py
# Acesse: http://localhost:8501
```

### C.3 Modelos de Previsão Implementados

#### C.3.1 ETS (Suavização Exponencial)
```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing
model = ExponentialSmoothing(series, seasonal='add', seasonal_periods=7)
```
- Captura tendência e sazonalidade semanal
- Funciona bem para séries regulares

#### C.3.2 LightGBM (Machine Learning)
Features utilizadas:
- **Lag features:** lag_1, lag_7, lag_14, lag_28
- **Médias móveis:** rolling_mean_7, rolling_mean_14, rolling_mean_28
- **Estatísticas de janela:** rolling_std, rolling_min, rolling_max
- **Tendência:** diff_1, diff_7
- **Cíclicas:** day_of_week (mod 7), week_of_month

Hiperparâmetros:
```python
lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8
)
```

#### C.3.3 Ensemble (Combinado)
Combina 3 modelos com pesos:
- ETS: 25%
- ARIMA: 25%
- LightGBM: 50%

A previsão final é a média ponderada:
```
ŷ = 0.25×ŷ_ETS + 0.25×ŷ_ARIMA + 0.50×ŷ_LGBM
```

#### C.3.4 Decomposição Sazonal
1. Decomposição aditiva (período=7)
2. Extração do padrão sazonal (último ciclo)
3. Regressão linear na tendência
4. Projeção: tendência futura + sazonalidade cíclica

### C.4 Métricas de Avaliação

| Métrica | Fórmula | Interpretação |
|---------|---------|---------------|
| **MAE** | \(\frac{1}{n}\sum|y_i - \hat{y}_i|\) | Erro absoluto médio (mesma escala) |
| **RMSE** | \(\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}\) | Penaliza erros grandes |
| **WAPE** | \(\frac{\sum|y_i - \hat{y}_i|}{\sum|y_i|}\) | Robusto para escalas diferentes |
| **SMAPE** | \(\frac{100}{n}\sum\frac{2|y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i|}\) | Simétrico, 0-200% |
| **Acurácia** | \(100\% - WAPE\) | Métrica intuitiva de assertividade |

### C.5 Política (R,Q) — Estoque de Segurança

**Fórmulas implementadas:**
```python
from scipy.stats import norm

# Parâmetros
L = lead_time_days
CSL = csl_target
μ_daily = forecast_mean
σ_daily = forecast_sigma

# Cálculos
z = norm.ppf(CSL)           # z-score para o CSL
μ_L = μ_daily * L           # Demanda média durante lead time
σ_L = σ_daily * sqrt(L)     # Desvio padrão durante lead time
SS = z * σ_L                # Estoque de segurança
R = μ_L + SS                # Ponto de ressuprimento
```

**Exemplo numérico:**
- μ_daily = 10 un/dia
- σ_daily = 3 un/dia
- L = 5 dias
- CSL = 95% → z = 1.645

Resultado:
- μ_L = 50 un
- σ_L = 6.71 un
- SS = 11.03 un
- R = 61.03 un

### C.6 Resultados Obtidos

**Classificação de Intermitência (50 itens analisados):**
- Errático: 21 (42%)
- Suave: 18 (36%)
- Grumoso: 11 (22%)

**Performance de Previsão (horizonte 28 dias, média dos itens):**

| Modelo | WAPE | Acurácia |
|--------|------|----------|
| LightGBM | 12-25% | 75-88% |
| Ensemble | 15-28% | 72-85% |
| ETS | 18-40% | 60-82% |
| Decomposição | 20-45% | 55-80% |

> Nota: Performance varia significativamente por item. Itens com alta intermitência (>50% zeros) têm WAPE maior.

### C.7 Execução do Pipeline

**Pré-requisitos:**
```bash
# Criar ambiente
conda create -n logistica python=3.11
conda activate logistica

# Instalar dependências
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
statsmodels>=0.14.0
scikit-learn>=1.3.0
scipy>=1.11.0
tqdm>=4.65.0
jupyter>=1.0.0
streamlit>=1.30.0
plotly>=5.18.0
lightgbm>=4.0.0
```

**Execução:**
```bash
# Pipeline completo (terminal)
python main.py

# Dashboard interativo
streamlit run dashboard.py

# Notebooks individuais
jupyter notebook notebooks/01_preprocess.ipynb
```

### C.8 Implementação da Etapa 2 (Simulação e Otimização)

A Etapa 2 foi **completamente implementada** com os seguintes módulos:

---

#### C.8.1 Simulação com SimPy (`src/simulation/simpy_env.py`)

**Ambiente de simulação com lead time estocástico:**

```python
# Configuração
config = SimulationConfig(
    horizon_days=365,
    n_replications=30,
    demand_distribution="normal",
    lead_time_distribution="truncnorm",  # Normal truncada
    lead_time_mean=5.0,
    lead_time_std=1.5,
    lead_time_min=2.0,
    lead_time_max=10.0
)
```

**Três políticas de estoque implementadas:**

| Política | Fórmula | Descrição |
|----------|---------|-----------|
| **(R, Q)** | Quando IP ≤ R, pedir Q | Revisão contínua, lote fixo |
| **(s, S)** | Quando IP ≤ s, pedir até S | Min-Max, lote variável |
| **(P, S)** | A cada P dias, pedir até S | Revisão periódica |

**Monte Carlo com intervalos de confiança:**
```python
# Executar N replicações e calcular IC 95%
results = run_monte_carlo(config, policy, n_replications=30)
# Saída: custo_mean, custo_std, custo_ci_lower, custo_ci_upper
```

**Métricas coletadas:**
- Custo total (pedido + holding + falta)
- Fill rate (% demanda atendida)
- Estoque médio e máximo
- Número de pedidos e faltas

---

#### C.8.2 Risk Pooling Avançado (`src/inventory/pooling_advanced.py`)

**Matriz de correlação entre lojas:**
```python
pivot, corr_matrix = calculate_correlation_matrix(df, store_ids)
# corr_matrix: DataFrame com ρ_ij para todas as lojas
```

**Cálculo de variância pooled com correlação:**
\[
\sigma^2_{pooled} = \sum_i \sigma^2_i + 2 \sum_{i<j} \rho_{ij} \sigma_i \sigma_j
\]

**Redução de SS centralizado vs descentralizado:**
```python
result = calculate_ss_reduction(df, stores, lead_time, csl)
# result: {ss_decentralized, ss_centralized, ss_reduction_pct}
```

**Cenários híbridos ABC:**
- **Classe A** (80% valor): Descentralizado (resposta rápida)
- **Classes B/C** (20% valor): Centralizado (reduz SS via pooling)

---

#### C.8.3 Otimização Multi-Item (`src/inventory/optimization.py`)

**Q* ótimo com custo de falta:**
\[
Q^* = \sqrt{\frac{2DK}{h}} \times \sqrt{\frac{h + p}{p}}
\]

**CSL ótimo endógeno:**
\[
CSL^* = 1 - \frac{h \times Q}{p \times D}
\]

**Multi-item com restrição de orçamento:**
```python
result = multi_item_budget_constraint(
    items,
    K=100, h_pct=0.20, p=50,
    budget=50000,
    csl_target=0.95
)
# Aloca Q e SS para cada item respeitando orçamento
```

**Modelo Newsvendor (perecíveis):**
```python
result = newsvendor_optimal_Q(
    D_mean, D_std,
    unit_cost, selling_price, salvage_value
)
# Q* = F^(-1)(Cu / (Cu + Co))
```

---

#### C.8.4 Dashboard - Página "Simulação & Etapa 2"

Nova página com 3 abas:

| Aba | Funcionalidade |
|-----|----------------|
| **🎰 Simulação Monte Carlo** | Compara políticas (R,Q), (s,S), (P,S) com lead time estocástico |
| **🔗 Risk Pooling Avançado** | Matriz de correlação, redução de SS, análise ABC híbrida |
| **📊 Otimização Multi-Item** | Q* com custo de falta, sensibilidade, alocação com orçamento |

---

#### C.8.5 Resultados da Etapa 2

**Simulação Monte Carlo (exemplo, 30 replicações):**

| Política | Custo Total | Fill Rate | Recomendação |
|----------|-------------|-----------|--------------|
| (R, Q) | R$ 15.230 | 96.8% | ✅ Melhor custo-benefício |
| (s, S) | R$ 15.890 | 97.2% | Maior fill rate |
| (P, S) | R$ 16.450 | 95.5% | Menor complexidade |

**Risk Pooling (3 lojas, correlação média = 0.45):**
- SS Descentralizado: 1.520 un
- SS Centralizado: 980 un
- **Redução: 35.5%**

**Otimização Multi-Item:**
- Com p/h = 10: CSL ótimo = 92%
- Com p/h = 50: CSL ótimo = 98%
- Orçamento R$ 50.000 → 10 itens otimizados

---

### C.9 Bibliotecas Instaladas (Etapa 2)

```
simpy>=4.0.0      # Simulação de eventos discretos
pulp>=2.7.0       # Programação linear/inteira mista
```

**Execução do Dashboard:**
```bash
conda activate logistica
pip install -r requirements.txt
streamlit run dashboard.py
# Acesse: http://localhost:8501 → Página "🎲 Simulação & Etapa 2"
```

---

### C.10 Conclusões e Próximos Passos

**Implementado nesta entrega:**
- [x] Simulação SimPy com lead time estocástico
- [x] Comparação de 3 políticas via Monte Carlo
- [x] Risk Pooling com correlação medida
- [x] Cenários ABC híbridos
- [x] Otimização Q* com custo de falta
- [x] Multi-item com orçamento
- [x] Dashboard interativo completo

**Próximos passos sugeridos:**
1. Backtest com janela deslizante em dados reais
2. Integração com previsão adaptativa
3. Otimização MILP com PuLP para casos grandes
4. Deploy do dashboard em servidor