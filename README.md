# 🧠 RAG Demo com PDF + Cohere + Pinecone + Groq

---

## 📌 O que é isso?

Este projeto é uma demonstração prática de **RAG (Retrieval-Augmented Generation)** usando:

- **PDFs como fonte de dados**
- **Embeddings com Cohere**
- **Armazenamento vetorial com Pinecone**
- **LLM da Groq para responder perguntas**

> 📚 Você envia um PDF → Ele é dividido em partes → Embeddings são gerados → Consultas são feitas com inteligência contextual.

---

## 💠 Tecnologias Usadas

- 🧱 **LangChain**
- 🧠 **Cohere API (embeddings)**
- 📦 **Pinecone (vector store)**
- 🤖 **Groq (LLM via LangChain)**
- 🌿 **dotenv (variáveis de ambiente)**
- 🐍 **Python 3.10+**

---

## 📂 Estrutura do Projeto

```
📁 raiz/
👥│
📂 data/
│   └── 2210.03629v3.pdf        # PDF usado para a demo
├── main.py                     # Script principal
├── .env                        # Suas chaves (NÃO suba isso pro Git)
└── requirements.txt            # Dependências
```

---

## ⚙️ Instalação Passo a Passo

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/seu-repo.git
cd seu-repo
```

### 2. Crie um ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Crie o arquivo `.env`

```
COHERE_API_KEY=your-cohere-api-key
PINECONE_API_KEY=your-pinecone-api-key
GROQ_API_KEY=your-groq-api-key
```

---

## 🚀 Como Rodar

1. Adicione um PDF na pasta `data/`.
2. Altere o caminho do arquivo no script, se necessário:

```python
PATH_FILE = "data\\2210.03629v3.pdf"
```

3. Execute:

```bash
python main.py
```

---

## 💡 O que acontece no script?

```text
1. Carrega o PDF
2. Divide em pedaços (chunks)
3. Gera embeddings via Cohere
4. Salva os vetores no Pinecone
5. Usa Groq para responder com contexto
```

---

## 🧪 Exemplo de Execução

```bash
Carregando os documentos ...
splitting the documments into small chunks ...

total chunks: 12

Resposta:
A large language model (LLM) is an advanced AI model trained on vast text data...
```

---

## 🛠️ Customizações

Quer mudar a pergunta?

```python
res = qa.invoke({"input": "what is a llm?"})
```

Quer trocar o PDF? É só substituir o arquivo na pasta `data/`.

---

## ⚠️ Observações

- Crie o **índice Pinecone** chamado `rag-demo` antes de rodar.
- As APIs da Cohere, Pinecone e Groq podem ter limitações ou custos.

---

## 🤝 Contribuindo

Curtiu o projeto? Tem ideia pra melhorar? Manda um PR, issue, ou só diz um oi 👋

---

## 📄 Licença

MIT — use livremente, só lembre de dar os créditos 😊

