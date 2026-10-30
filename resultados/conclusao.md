# Conclusão da execução

Foram treinados YOLOv8n e YOLO11n por 30 épocas cada, com imagens de
640 pixels, batch 16 e seed 42, a partir de pesos pré-treinados em COCO.
Após a auditoria, foram usadas 1132 imagens de treino,
143 de validação e 141 de teste.
Os checkpoints foram selecionados na validação, antes da avaliação no teste.

O YOLOv8n obteve mAP50 de 0.5446 e mAP50-95 de 0.2766.
O YOLO11n obteve mAP50 de 0.5425 e mAP50-95 de 0.2741.
YOLOv8n obteve o maior mAP50-95, com vantagem de 0.25 pontos percentuais.

Na GPU Tesla T4, a etapa de inferência medida pelo validador levou
16.05 ms/imagem para YOLOv8n e 14.41
ms/imagem para YOLO11n, com batch 1 e FP32, sem pré/pós-processamento.
Esses tempos são indicativos desta execução, sem repetições de benchmark.

As imagens com boxes permitem confrontar as previsões com as anotações de referência.
O AP por classe mostra diferenças que a média geral pode esconder.
Foi feita uma execução por versão; não foram estimadas variância ou significância
estatística. O experimento se limita à base Construction-PPE e às suas anotações.
