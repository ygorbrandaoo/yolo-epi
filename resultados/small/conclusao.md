# Conclusão da execução

Foram treinados YOLOv8s e YOLO11s com limite de 50 épocas por modelo,
com imagens de
640 pixels, batch 16 e seed 42, a partir de pesos pré-treinados em COCO.
Foram realizadas 50 épocas para YOLOv8s e
50 épocas para YOLO11s. O parâmetro PATIENCE foi 10.
Após a auditoria, foram usadas 1132 imagens de treino,
143 de validação e 141 de teste.
Os checkpoints foram selecionados na validação, antes da avaliação no teste.

O YOLOv8s obteve mAP50 de 0.5329 e mAP50-95 de 0.2708.
O YOLO11s obteve mAP50 de 0.5585 e mAP50-95 de 0.2889.
YOLO11s obteve o maior mAP50-95, com vantagem de 1.81 pontos percentuais.

O macro-F1 foi 0.5353 para YOLOv8s e 0.5646
para YOLO11s, calculado pela média dos F1 por classe, com o mesmo peso
para cada classe presente no teste. As tabelas exportadas incluem o F1 de cada classe.

Na GPU Tesla T4, a etapa de inferência medida pelo validador levou
18.58 ms/imagem para YOLOv8s e 19.03
ms/imagem para YOLO11s, com batch 1 e FP32, sem pré/pós-processamento.
Esses tempos são indicativos desta execução, sem repetições de benchmark.

As imagens com boxes permitem confrontar as previsões com as anotações de referência.
O AP por classe mostra diferenças que a média geral pode esconder.
Foi feita uma execução por versão; não foram estimadas variância ou significância
estatística. O experimento se limita à base Construction-PPE e às suas anotações.
