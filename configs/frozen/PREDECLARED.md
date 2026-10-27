# Regras pré-declaradas da avaliação no teste do PoseLift

Escrito na sessão 3 (26/10/2026), antes de qualquer abertura do teste, com as decisões do grupo P13, P15, P16 e P17 (`docs/tcc/sessoes/SESSAO_3.md`). Este arquivo e as configurações desta pasta são commitados antes do primeiro `--final` e não mudam depois (ADR-0009). Qualquer ajuste exige uma configuração congelada nova, e a nova abertura do teste é contada.

## 1. Como o teste é aberto

- Só por `scripts/evaluate.py --config configs/frozen/<nome>.yaml --final`, com autorização explícita do usuário na sessão.
- O script recusa a abertura se a configuração não estiver commitada nesta pasta, se este arquivo não estiver commitado ou se a árvore tiver qualquer alteração.
- Antes de ler qualquer vídeo de teste, ele grava em `runs/test_evaluations.csv`: data, run, configuração, sha256 do arquivo, blob do git, sha256 deste arquivo, commit e o número da abertura.
- O sha256 de cada checkpoint está na configuração congelada, e a avaliação recusa um checkpoint diferente.
- **Aberturas planejadas: cinco**, uma por configuração (`main`, `nan_zero`, `t12`, `t48`, `gate_off`). O número real de aberturas é reportado na monografia.

## 2. Configurações congeladas

| Arquivo | O que é | Sementes |
|---|---|---|
| `main.yaml` | Principal: filtro de qualidade e proteção da escala ligados, T = 24, NaN interpolado | 0 a 4 |
| `nan_zero.yaml` | Sensibilidade: confiança NaN = 0 | 0 |
| `t12.yaml` | Sensibilidade: T = 12, com os limiares do filtro medidos com T = 24 | 0 |
| `t48.yaml` | Sensibilidade: T = 48, com os limiares do filtro medidos com T = 24 | 0 |
| `gate_off.yaml` | Sensibilidade: `quality_gate: off` (proteção da escala ligada) | 0 |

A oclusão de 10% a 50% é aplicada no teste sobre o modelo principal, dentro de `main.yaml`.

Nenhuma das sensibilidades é escolhida pelo resultado. Todas são reportadas.

## 3. Limiar e pontos de operação (decisão 13)

- **Principal: N = 5 alertas por hora.** Secundários: N = 2 e N = 10.
- **Regra, igual para cada modelo** (cada semente e cada configuração calcula o próprio limiar na própria validação):
  - fluxo operacional da validação: escore causal no último quadro da janela, máximo entre pessoas, filtro de qualidade, sem suavização;
  - candidatos: os quantis únicos, em 1.000 níveis de 0,5 a 1,0, dos escores dos quadros pontuados;
  - limiar: o menor candidato com no máximo N alertas por hora, contando todos os quadros da validação;
  - lógica de alerta: 15 quadros pontuados seguidos acima do limiar (`escore > limiar`), fusão de alertas a menos de 30 quadros.
- **O N é um orçamento de revisão declarado, não uma taxa medida.** A validação tem 14,8 min, e um alerta nela equivale a 4,06 alertas por hora. Zero alertas é compatível, a 95% (Poisson), com até cerca de 12 alertas por hora.

## 4. Métricas e protocolos (decisões 15 e 16)

### 4.1. Protocolo offline do STG-NF (ADR-0008), para AUC-ROC, AUC-PR e EER

- **Atribuição:** o escore da janela vai para o centro da janela (primeiro quadro + T/2); o escore do quadro é o máximo entre as pessoas.
- **Quadros avaliados:** todos os quadros do rótulo.
- **Quadros sem escore:** sem pessoa, ou só com janelas "não pontuáveis". Eles recebem o menor escore finito do conjunto avaliado.
- **Suavização:** gaussiana aplicada uma vez para cada sigma de 1 a 6, em sequência.
- **Métricas:**

| Métrica | Definição |
|---|---|
| AUC-ROC | `roc_auc_score`, furto positivo |
| **AUC-PR principal** | precisão média (`average_precision_score`), **furto positivo** |
| AUC-PR comparável à Tabela 3 | trapézio sobre a curva precisão–revocação (`auc(recall, precision)`), **normal positivo** com o escore negado, como no código do STG-NF |
| EER | média da taxa de falsos positivos e da de falsos negativos no ponto da ROC em que estão mais próximas |

- **Número otimista:** o centro da janela usa 0,8 s de quadros futuros, então este número é otimista em relação à operação.
- **Diferença no tratamento:** a regra de escore reproduz o código distribuído (teste de caracterização), mas as janelas não são as mesmas. O nosso tratamento gera também a última janela de cada trajetória e interpola lacunas até k = 8.

### 4.2. Fluxo operacional, para as métricas no limiar

- **Escore:** causal no último quadro, com filtro de qualidade, sem suavização, comparado ao limiar da seção 3 com `escore > limiar`.
- **Por quadro, sobre todos os quadros do teste** (quadro sem escore = negativo):
  - precisão, revocação, especificidade e F1;
  - **H-PRS** = 3 / (1/P + 1/R + 1/S), com S = 1 − FPR;
  - matriz de confusão.
- **Situação do H-PRS:** a definição foi conferida na fonte primária, o RetailS (arXiv 2603.04723v1, Seç. III-B2), que a apresenta sem citar fonte anterior. A conferência foi por leitura automatizada da versão HTML, duas vezes. Por isso o H-PRS entra.
- **Eventos:**
  - um evento é um trecho contínuo de quadros com rótulo 1 num vídeo;
  - um alerta acerta o evento se o início do alerta cair durante o evento ou até 30 quadros (2 s) depois do fim;
  - o início do alerta é o primeiro quadro do trecho acima do limiar, não o quadro em que o alerta dispara (15 quadros depois).
- **Medidas de evento:**
  - revocação de eventos (eventos acertados / eventos);
  - precisão de alertas (alertas que acertam algum evento / alertas);
  - falsos alertas por hora (alertas que não acertam nenhum evento, por hora de teste);
  - alertas por hora.
- **Ruído:** o teste tem poucos minutos de vídeo (cerca de 4 min de quadros com pessoa), e as taxas por hora no teste são muito ruidosas.
- **Pontos de operação:** todas as métricas desta seção são reportadas para N = 5 (principal) e N = 2 e N = 10 (secundários).

### 4.3. Descritivas (ADR-0014)

- **Quadros não pontuáveis:** a fração dos quadros do teste "não pontuáveis", por motivo do filtro. Ela é reportada nas duas atribuições: centro (a que as AUC veem) e último quadro (a que os alertas veem).
- **Ponto cego do filtro:** a fração dos quadros de furto que estão dentro de pelo menos uma janela filtrada.

## 5. Sementes e intervalo de confiança (decisão 17)

- **Agregação:** cada métrica é reportada por semente e como média e desvio-padrão amostral (n − 1) das 5 sementes do principal.
- **Bootstrap da AUC-ROC:**
  - 2.000 reamostras, semente 0, intervalo de 95% pelo percentil;
  - reamostragem por vídeo, estratificada: vídeos com furto e vídeos só normais são sorteados separadamente, com reposição, mantendo as contagens;
  - em cada reamostra, a AUC-ROC de cada semente é calculada sobre os mesmos vídeos e tirada a média;
  - o intervalo é o dessa média. Nas sensibilidades (uma semente), o intervalo é o da semente única.
- **Variação do treino:** o treino na GPU não é reprodutível bit a bit, e essa variação entra no desvio entre sementes (decisão 18). A inferência a partir do checkpoint é reprodutível e roda em `cuda`, como na escolha dos limiares.

## 6. Oclusão (P13, item 3)

- **Frações:** 10%, 20%, 30%, 40% e 50% das articulações perdem a confiança (zerada), no teste, sobre as 5 sementes do principal.
- **Ordem no tratamento:** a oclusão é aplicada depois da interpolação e antes da normalização e do filtro. Assim o filtro vê a pose ocluída, como veria na operação.
- **Máscara:** é a mesma para todas as sementes (gerador com semente `[0, fração × 1000, índice do vídeo]`).
- **Limiar:** o de cada semente, da validação sem oclusão.
- **Métricas:** as mesmas das seções 4.1 a 4.3.

## 7. Renderizações depois do `--final` (P16, item 12)

- **Ferramenta:** `python -m scripts.render_final`, com `configs/render_final.yaml`, apontando para o run de `--final` de `main.yaml`.
- **O que é renderizado:** todos os vídeos de teste com furto anotado, em ordem alfabética, com:
  - esqueleto (só esqueletos; o PoseLift não tem vídeo bruto);
  - linha do tempo com o escore operacional da semente 0 no N principal;
  - limiar, alertas, intervalos anotados e trechos "não pontuáveis" por motivo.
- **Slides:** os dois primeiros em ordem alfabética. Nenhuma escolha olhando o resultado.

## 8. Referências de comparação

- **Tabela 3 do PoseLift** (AUC-ROC %, AUC-PR %, EER): STG-NF 67,46 / 84,06 / 0,39; TSGAD 63,35 / 39,31 / 0,41; GEPC 60,61 / 50,38 / 0,38. A AUC-PR da Tabela 3 provavelmente usa normal positivo (inferência registrada), por isso a coluna da seção 4.1.
- **PoC de Luiz:** AUC-ROC 64,34% no teste, número preliminar e não rastreável (semente, configuração e número de avaliações desconhecidos).

## 9. Depois da abertura

Nenhum parâmetro muda depois do `--final`: limiar, N, sigmas, filtro, T e sementes ficam como aqui.
