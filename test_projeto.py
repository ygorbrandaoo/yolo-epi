import tempfile
import unittest
import json
import base64
import csv
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


if __name__ == '__main__':
    unittest.main()
