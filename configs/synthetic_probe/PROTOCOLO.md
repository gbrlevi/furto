# Protocolo da sondagem sintética (pré-declarado)

Decidido por Gabriel em 01/11/2026 (P20). Este arquivo e o `prompts.yaml` ao lado são congelados **antes** de gerar qualquer vídeo. Depois do congelamento, nada aqui muda. Desvios viram uma linha no relatório, nunca uma edição deste arquivo.

## 1. Objetivo e limite

**Pergunta.** Com a configuração congelada `main`, o pipeline reage de forma diferente a vídeos sintéticos de ocultação de produto e a vídeos sintéticos de compra normal, gerados pelo mesmo gerador, no mesmo cenário e com os mesmos movimentos amplos?

**O que a sondagem é.** Uma ilustração qualitativa, em domínio sintético.

**O que ela não é:**

- não é métrica e não entra nas tabelas de resultado;
- não muda nenhum parâmetro, limiar ou configuração;
- não substitui a avaliação no PoseLift (27/10/2026), que continua sendo o resultado do trabalho.

**Por que há pares.** O autoencoder reage a qualquer pose fora da distribuição do treino (achado da 2C), inclusive a artefatos de geração. Sem um vídeo normal do mesmo gerador ao lado, um alerta no vídeo de ocultação não distingue "o modelo viu a ocultação" de "o modelo viu um corpo gerado".

## 2. Geração (feita por um membro do grupo)

1. Preencha o bloco `generator` do `prompts.yaml` (nome, versão, resolução, semente) e faça o congelamento dos dois arquivos.
2. Gere os 10 vídeos na ordem de `order`, cada um com o prompt final abaixo:
   - prompt final = `style` + "The person is " + `person` + ". " + o texto da variante (`concealment` ou `normal`);
   - use o `negative_prompt`, se o gerador aceitar.
3. **Uma geração por prompt.**
   - Não regere porque o vídeo "não ficou bom" ou porque a ação não aparece.
   - A única exceção é a **falha técnica**: arquivo corrompido, duração menor que 8 s, mais de uma pessoa em cena durante a maior parte do vídeo, ou recusa do gerador.
   - Na falha técnica, gere de novo uma vez, com o mesmo prompt, e registre o motivo. Se falhar outra vez, o vídeo entra como "não gerado".
   - Se o gerador recusar um prompt, registre a recusa. Não reescreva o prompt.
4. Salve os vídeos em `data/raw/synthetic/<id>.mp4`, fora do git.
5. Preencha `data/raw/synthetic/manifest.csv` com: id, data e hora da geração, gerador, versão, configurações, número de tentativas e motivo de cada nova tentativa. As datas de geração devem ser posteriores ao congelamento do item 1.

## 3. Anotação, antes de rodar o pipeline

Dois membros do grupo, cada um sem ver o trabalho do outro e sem ver nenhum escore, anotam cada um dos 10 vídeos:

| Campo | Valores |
|---|---|
| A ação pedida aparece? | sim / não / ambíguo |
| Intervalo da ocultação, em segundos | início e fim, ou "nenhum" |
| Artefatos visíveis no corpo (membros deformados, corpo que muda, partes que somem) | nenhum / leves / graves |

- A anotação vai para `data/raw/synthetic/annotations_<nome>.csv`.
- **Intervalo de referência:** a união dos dois intervalos anotados.
- **Vídeos de ocultação sem a ação** ("não" ou "ambíguo"): continuam na tabela, marcados. Não saem.

## 4. Processamento

- **Configuração:** `configs/frozen/main.yaml`, sem nenhuma alteração.
- **Principal:** o pipeline online da 3B (MMPose, flip ligado, filtro de qualidade, proteção da escala), com o checkpoint da semente 0 e o limiar de N = 5 dela (0,266779). É o que seria mostrado na apresentação.
- **Complementar:** o mesmo vídeo pontuado offline com as sementes 1 a 4, cada uma com o próprio limiar de N = 5, para que o resultado não dependa da semente escolhida.
- **Reamostragem:** 15 fps, como em produção.
- **Regra de alerta:** a do `PREDECLARED.md` — 15 quadros consecutivos acima do limiar e fusão a menos de 2 s.
- **Acerto:** o quadro do disparo cai dentro do intervalo de referência ou até 30 quadros depois do fim dele.

## 5. O que se reporta, para os 10 vídeos, sem exceção

| Coluna | Conteúdo |
|---|---|
| id | p1_concealment … p5_normal |
| Ação apareceu / artefatos | da anotação |
| Quadros com pessoa pontuável (%) | e o motivo mais frequente de "não pontuável" |
| Maior escore causal / limiar | razão, semente 0 |
| Alertas (semente 0) | número e se acertou o intervalo de referência |
| Alertas nas sementes 1 a 4 | quantas das 4 alertaram |

**Figuras:**

- linha do tempo de cada vídeo, com escore, limiar, alertas, intervalo anotado e trechos não pontuáveis;
- vídeo só com esqueleto do par `p1` (os dois vídeos), escolhido antes de ver qualquer resultado.

## 6. Leituras declaradas antes de rodar

| Resultado (semente 0, vídeos em que a ação apareceu) | Leitura |
|---|---|
| Ocultação alerta, normal do mesmo par não alerta, na maioria dos pares | Sinal qualitativo compatível com a hipótese, em domínio sintético; não é métrica |
| Os dois vídeos do par alertam | O modelo reage ao corpo gerado ou ao movimento amplo, não à ocultação (mesma confusão da 2C) |
| Nenhum dos dois alerta | Coerente com o teste do PoseLift: o limiar de N = 5 é conservador |
| A maioria dos quadros fica "não pontuável" | O vídeo sintético fica fora do domínio operacional do filtro; a sondagem não informa sobre o modelo |
| Pares divididos entre as leituras acima | Reportar par a par, sem resumir num único veredito |

## 7. Uso na apresentação e na monografia

- O slide traz a tabela inteira dos 10 vídeos, o par `p1` e a leitura que couber.
- O rótulo fica visível no slide: **"Vídeos sintéticos gerados por IA (gerador, versão). Ilustração qualitativa; não é avaliação."**
- O uso de IA generativa é mencionado como a política do curso exige.
- Na monografia, a sondagem entra em "Resultados e discussões" como ilustração, depois dos resultados do PoseLift, nunca antes ou no lugar deles.

## 8. Ética

- As pessoas são inteiramente sintéticas e adultas; nenhum prompt descreve pessoa real, marca ou loja real.
- Os vídeos só servem a este protocolo e não entram em treino nem em validação.