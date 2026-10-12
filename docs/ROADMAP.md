# Roadmap da prova de conceito

O plano tem 10 etapas, aprovadas pelo grupo em 12/10/2026. Cada etapa só está pronta quando todos os seus critérios de aceite têm teste ou medição registrada.

**Escopo do PFI1 (prazo por volta de 11/11/2026).** O PFI1 cobre as etapas 1 a 7 e a parte "PFI1" da etapa 9: o pipeline em arquivo de vídeo até "escore > limiar". As etapas 8 e 10 ficam para o PFI2.

**Porte do PoC de Luiz.** As etapas 3 a 6 são feitas portando o código dele, conforme o ADR-0006 e `docs/tcc/PORTE_POC_LUIZ.md`. Os critérios de aceite abaixo não mudam: código portado sem teste não conta como pronto.

**Semanas.** A divisão abaixo ainda é proposta, a confirmar pelo grupo. Foi revista em 12/10/2026 por causa do porte.

| Semana | Período | Etapas |
| --- | --- | --- |
| 1 | 12 a 18/10 | Sessão 1: etapas 1 e 2, mais o porte do esqueleto e da fixture sintética. Em paralelo: instalação da pilha de pose (início da etapa 7) |
| 2 | 19 a 25/10 | Sessão 2: run de referência com o código original (sem tocar no teste), porte das etapas 3 e 5 com equivalência e primeiro treino C1 (só validação). Em paralelo: extrator de pose sobre o MEVA (etapa 7) |
| 3 | 26/10 a 01/11 | Sessão 3: porte das etapas 4 e 6 com o portão do teste. Depois: fila de T quadros ligada ao modelo |
| 4 | 02 a 08/11 | Etapa 9, parte PFI1; C1 final com cinco sementes; medições no PC |
| Folga | 09 a 11/11 | — |

## 1. Fundação

- `just check` verde num repositório que tem ao menos um teste.
- O `run_logger` grava `meta.json` com todos os campos da regra 4 do CLAUDE.md. Um teste verifica cada campo, inclusive `dirty`.
- O `.gitignore` cobre `data/`, `runs/`, vídeos, pesos e `CLAUDE.md`.

## 2. Dados do PoseLift

- Licença e termos de uso registrados com o link. Se forem restritivos, parar.
- `docs/tcc/FORMATO_POSELIFT.md` escrito.
- Contagens por partição medidas e comparadas à Tabela 1 do CONTEXTO_TCC. Ou batem, ou a divergência está registrada.
- O carregador lê treino e teste no contrato do ADR-0004.
- Configuração de extração do PoseLift registrada, ou registrada como não encontrada.
- O visualizador gera um mp4 de esqueleto.
- Distribuição do comprimento das trajetórias medida.

## 3. Tratamento

Cada item abaixo tem seu próprio teste:

- interpolação só até k quadros;
- descarte de trajetórias com menos de T quadros;
- zeragem abaixo do limiar de confiança;
- normalização invariante a translação e escala;
- espelhamento que troca os pares esquerda/direita;
- janelas corretas com T e S nos dois modos (treino e avaliação);
- divisão por vídeo sem nenhum vídeo em comum entre treino e validação.

## 4. Harness de avaliação

- Escore aleatório dá AUC-ROC ≈ 0,5, dentro de uma tolerância declarada no teste.
- Escore oráculo dá 1,0.
- Atribuição janela → quadro, máximo entre pessoas e suavização testados.
- Regra para quadros sem ninguém em cena documentada e conferida contra o protocolo do PoseLift.
- Se houver código público de um baseline da Tabela 3, a tentativa de reprodução fica registrada.

## 5. Autoencoder ST-GCN (C1)

- Teste de sobreajuste num único lote: a perda cai perto de zero.
- Com o checkpoint do run de referência próprio (código original, semente 0) carregado, o modelo portado reproduz as pontuações da validação do original dentro de tolerância numérica (`docs/tcc/PORTE_POC_LUIZ.md`).
- Treino C1 com uma semente, com run registrado.
- A configuração é congelada antes de abrir o teste.
- Métricas C1 no teste calculadas uma vez por configuração congelada. Depois disso, nenhum ajuste.

## 6. Limiar e lógica de alerta

- **PFI1:**
  - limiar provisório derivado só da validação, com o método documentado;
  - lógica de alerta simulada na validação, com alertas por hora medidos.
- **Depois:** limiar pelo orçamento de N alertas por hora, com N fixado pelo grupo.

## 7. Extração de pose a partir de vídeo

- Um arquivo de vídeo vira registros no contrato do ADR-0004.
- Variantes, pesos e limiares iguais aos da extração do PoseLift. Se diferirem, ADR Proposto explicando.
- Comparação de distribuições com o PoseLift registrada: confiança por articulação, comprimento das trajetórias, proporções de ossos.
- Fps da extração medido no PC de desenvolvimento.

## 8. Backend de triagem (fora do PFI1)

- Buffer circular de 60 s do fluxo comprimido.
- Clipe de 10 s antes a 5 s depois do alerta.
- Registro auditável com hash do clipe, escore, limiar, versão do modelo e decisão.
- Interface mínima do operador.
- RTSP por último.

## 9. Integração e custo

- **PFI1:**
  - pipeline ponta a ponta sobre arquivo de vídeo emite escore por pessoa e evento "escore > limiar" num log;
  - fps e latência (média e p95) medidos no PC de desenvolvimento.
- **Depois:** medições no hardware de referência e com vídeo do conjunto encenado.

## 10. RetailS, robustez e sensibilidade (fora do PFI1)

- C2 e C3.
- Remoção de 10% a 50% das articulações.
- Sensibilidade a T = 12, 24 e 48.
- Conjunto encenado depois da aprovação do CEP.
