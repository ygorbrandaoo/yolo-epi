import tempfile
import unittest
import json
import base64
import csv
import ast
from types import SimpleNamespace
from pathlib import Path

from auditoria import preparar_divisoes, ler_boxes, resumir_classes


class AuditoriaTest(unittest.TestCase):
    def test_converte_boxes_e_preserva_classe(self):
        with tempfile.TemporaryDirectory() as pasta:
            arquivo = Path(pasta) / 'a.txt'
            arquivo.write_text('2 0.5 0.4 0.2 0.3\n', encoding='utf-8')
            self.assertEqual(ler_boxes(arquivo, 11), [(2, 0.5, 0.4, 0.2, 0.3)])

    def test_anotacao_vazia_representa_imagem_sem_objetos(self):
        with tempfile.TemporaryDirectory() as pasta:
            arquivo = Path(pasta) / 'a.txt'
            arquivo.write_text('', encoding='utf-8')
            self.assertEqual(ler_boxes(arquivo, 11), [])

    def test_rejeita_anotacoes_invalidas(self):
        exemplos = ['11 0.5 0.5 0.2 0.2', '0 0.5 nan 0.2 0.2',
                    '0 1.2 0.5 0.2 0.2', '0 0.5 0.5 -0.2 0.2',
                    '0.5 0.5 0.5 0.2 0.2', '0 0.5 0.5 0.2']
        with tempfile.TemporaryDirectory() as pasta:
            arquivo = Path(pasta) / 'a.txt'
            for texto in exemplos:
                with self.subTest(texto=texto):
                    arquivo.write_text(texto, encoding='utf-8')
                    with self.assertRaises(ValueError):
                        ler_boxes(arquivo, 11)

    def test_anotacao_ausente_nao_e_ignorada(self):
        with tempfile.TemporaryDirectory() as pasta:
            with self.assertRaises(FileNotFoundError):
                ler_boxes(Path(pasta) / 'ausente.txt', 11)

    def test_remove_vazamento_por_conteudo_e_preserva_originais(self):
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            divisoes = {}
            for split in ['train', 'val', 'test']:
                imagens = raiz / 'images' / split
                labels = raiz / 'labels' / split
                imagens.mkdir(parents=True)
                labels.mkdir(parents=True)
                (imagens / 'unica.jpg').write_bytes(split.encode())
                (labels / 'unica.txt').write_text('0 0.5 0.5 0.2 0.2')
                divisoes[split] = imagens
            (divisoes['val'] / 'copia.jpg').write_bytes(b'train')
            (raiz / 'labels/val/copia.txt').write_text('0 0.5 0.5 0.2 0.2')
            listas, resumo, duplicatas, contagens = preparar_divisoes(divisoes, 2)
            self.assertEqual([len(listas[s]) for s in divisoes], [1, 1, 1])
            self.assertEqual(len(duplicatas), 1)
            self.assertEqual(resumo['val']['duplicatas_excluidas'], 1)
            self.assertTrue((divisoes['val'] / 'copia.jpg').exists())
            self.assertEqual(contagens['test'], [1, 0])

    def test_rejeita_conjunto_que_ficou_vazio(self):
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            with self.assertRaises(ValueError):
                preparar_divisoes({s: raiz for s in ['train', 'val', 'test']}, 11)

    def test_classe_sem_predicoes_recebe_zero_e_classe_ausente_e_omitida(self):
        linhas = resumir_classes('modelo', {0: 'helmet', 1: 'vest'}, [3, 0], [], [], [], [], [])
        self.assertEqual(len(linhas), 1)
        self.assertEqual(linhas[0]['classe'], 'helmet')
        self.assertEqual(linhas[0]['AP50-95'], 0)

    def test_metricas_por_classe_usam_indices_reais(self):
        linhas = resumir_classes('modelo', {0: 'helmet', 1: 'vest', 2: 'boots'},
                                 [2, 0, 4], [2, 0], [0.6, 0.8], [0.5, 0.7],
                                 [0.4, 0.6], [0.2, 0.3])
        self.assertEqual([r['classe'] for r in linhas], ['helmet', 'boots'])
        self.assertEqual([r['AP50-95'] for r in linhas], [0.3, 0.2])

    def test_f1_por_classe_e_zero_sem_predicoes(self):
        linhas = resumir_classes('modelo', {0: 'helmet', 1: 'boots'},
                                 [2, 3], [0], [0.5], [1.0], [0.4], [0.2])
        self.assertAlmostEqual(linhas[0]['f1'], 2 / 3)
        self.assertEqual(linhas[1]['f1'], 0)


class NotebookTest(unittest.TestCase):
    def test_notebook_inclui_metricas_e_imagens_reais_da_execucao(self):
        raiz = Path(__file__).resolve().parent
        notebook = json.loads((raiz / 'yolo_epi.ipynb').read_text(encoding='utf-8'))
        fontes = '\n'.join(c['source'] for c in notebook['cells'] if c['cell_type'] == 'markdown')
        with (raiz / 'resultados/comparacao.csv').open(encoding='utf-8') as arquivo:
            resultados = list(csv.DictReader(arquivo))
        for linha in resultados:
            self.assertIn(f"{float(linha['mAP50-95']) * 100:.2f}%", fontes)
        anexos = {nome: conteudo for c in notebook['cells']
                  for nome, conteudo in c.get('attachments', {}).items()}
        for i in range(1, 7):
            nome = f'comparacao_{i:02d}.jpg'
            self.assertIn(nome, anexos)
            self.assertEqual(base64.b64decode(anexos[nome]['image/jpeg']),
                             (raiz / 'resultados/visuais' / nome).read_bytes())

    def test_celulas_python_compilam_e_auditoria_esta_embutida(self):
        raiz = Path(__file__).resolve().parent
        notebook = json.loads((raiz / 'yolo_epi.ipynb').read_text(encoding='utf-8'))
        self.assertEqual(notebook['nbformat'], 4)
        fontes = [c['source'] for c in notebook['cells'] if c['cell_type'] == 'code']
        for indice, fonte in enumerate(fontes):
            compile(fonte, f'celula-{indice}', 'exec')
        self.assertIn((raiz / 'auditoria.py').read_text(encoding='utf-8').strip(),
                      [fonte.strip() for fonte in fontes])

    def test_auditoria_embutida_executa_sem_arquivos_externos(self):
        raiz = Path(__file__).resolve().parent
        notebook = json.loads((raiz / 'yolo_epi.ipynb').read_text(encoding='utf-8'))
        fonte = next(c['source'] for c in notebook['cells']
                     if c['cell_type'] == 'code' and 'def preparar_divisoes(' in c['source'])
        contexto = {}
        exec(compile(fonte, 'auditoria_no_notebook', 'exec'), contexto)
        linhas = contexto['resumir_classes']('teste', {0: 'helmet'}, [1], [], [], [], [], [])
        self.assertEqual(linhas[0]['AP50'], 0.0)


class ExperimentoTest(unittest.TestCase):
    def notebook(self):
        caminho = Path(__file__).resolve().parent / 'experimento.ipynb'
        return json.loads(caminho.read_text(encoding='utf-8'))

    def configurar(self, **valores):
        fonte = next(c['source'] for c in self.notebook()['cells'] if c['id'] == 'parametros')
        arvore = ast.parse(fonte)
        for node in arvore.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                nome = getattr(node.targets[0], 'id', None)
                if nome in valores:
                    node.value = ast.Constant(valores[nome])
        contexto = {}
        exec(compile(ast.fix_missing_locations(arvore), 'parametros', 'exec'), contexto)
        return contexto

    def test_configuracao_inicial_usa_small_e_cinquenta_epocas(self):
        contexto = self.configurar()
        self.assertEqual(contexto['MODELOS'], {'YOLOv8s': 'yolov8s.pt', 'YOLO11s': 'yolo11s.pt'})
        self.assertEqual(contexto['EPOCHS'], 50)
        self.assertEqual(contexto['IMGSZ'], 640)

    def test_parada_antecipada_esta_ativa_e_chega_ao_treinador(self):
        contexto = self.configurar()
        self.assertEqual(contexto['PATIENCE'], 10)
        contexto = self.configurar(PATIENCE=7)
        fonte = next(c['source'] for c in self.notebook()['cells']
                     if c['cell_type'] == 'code' and 'PARAMETROS = dict(' in c['source'])
        node = next(n for n in ast.parse(fonte).body if isinstance(n, ast.Assign)
                    and getattr(n.targets[0], 'id', None) == 'PARAMETROS')
        contexto['DEVICE'] = 0
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'treinador', 'exec'), contexto)
        self.assertEqual(contexto['PARAMETROS']['patience'], 7)
        self.assertEqual(contexto['PARAMETROS']['epochs'], 50)

    def test_tamanho_altera_os_dois_modelos(self):
        for tamanho in ['n', 's', 'm']:
            with self.subTest(tamanho=tamanho):
                contexto = self.configurar(TAMANHO=tamanho, EPOCHS=2, BATCH=8)
                self.assertEqual(list(contexto['MODELOS'].values()),
                                 [f'yolov8{tamanho}.pt', f'yolo11{tamanho}.pt'])
                self.assertEqual(contexto['BATCH'], 8)

    def test_parametros_invalidos_falham_antes_de_baixar_e_treinar(self):
        for valores in [{'TAMANHO': 'x'}, {'EPOCHS': 0}, {'BATCH': -1},
                        {'IMGSZ': 641}, {'SEED': -1}, {'LR0': 0},
                        {'CONF_VISUAL': 1.5}, {'PATIENCE': -1}, {'PATIENCE': 2.5}]:
            with self.subTest(valores=valores), self.assertRaises(ValueError):
                self.configurar(**valores)

    def test_conclusao_funciona_com_nomes_dos_modelos_escolhidos(self):
        fonte = next(c['source'] for c in self.notebook()['cells']
                     if c['cell_type'] == 'code' and c['source'].startswith('v8 = comparacao.set_index'))
        for tamanho in ['n', 's', 'm']:
            with self.subTest(tamanho=tamanho), tempfile.TemporaryDirectory() as pasta:
                contexto = self.configurar(TAMANHO=tamanho)
                resultado = {'mAP50': 0.5, 'mAP50-95': 0.3, 'macro_f1': 0.4,
                             'inferencia_ms': 10.0, 'epocas_realizadas': 37}
                contexto.update(comparacao=SimpleNamespace(set_index=lambda _: SimpleNamespace(
                    loc={nome: resultado for nome in contexto['MODELOS']})),
                    AMBIENTE={'gpu': 'teste'}, PASTA=Path(pasta),
                    LISTAS={'train': [1], 'val': [2], 'test': [3]},
                    CONTAGENS={'test': [1]}, NOMES={0: 'helmet'},
                    display=lambda _: None, Markdown=lambda texto: texto)
                exec(compile(fonte, 'conclusao', 'exec'), contexto)
                texto = (Path(pasta) / 'conclusao.md').read_text(encoding='utf-8')
                self.assertIn(f'YOLOv8{tamanho}', texto)
                self.assertIn(f'YOLO11{tamanho}', texto)
                self.assertIn('37', texto)
                self.assertIn('limite de 50', texto)

    def test_notebook_novo_tem_sintaxe_valida_sem_saidas_antigas(self):
        notebook = self.notebook()
        for celula in notebook['cells']:
            if celula['cell_type'] == 'code':
                compile(celula['source'], celula['id'], 'exec')
                self.assertFalse(celula['outputs'])
        self.assertFalse(any(c.get('attachments') for c in notebook['cells']))


if __name__ == '__main__':
    unittest.main()
