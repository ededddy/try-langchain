import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import { ChatGroq } from "@langchain/groq";
import { NomicEmbeddings } from "@langchain/nomic";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { pull } from "langchain/hub";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { formatDocumentsAsString } from "langchain/util/document";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

const pTagSelector = "p";
const loader = new CheerioWebBaseLoader(
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  {
    selector: pTagSelector,
  },
);

const docs = await loader.load();

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

const allSplits = await textSplitter.splitDocuments(docs);

const vecStore = await MemoryVectorStore.fromDocuments(
  allSplits,
  new NomicEmbeddings(),
);

const retriever = vecStore.asRetriever({
  k: 6,
  searchType: "similarity",
});

const llm = new ChatGroq({
  model: "mixtral-8x7b-32768",
  temperature: 0,
});

const prompt = await pull<ChatPromptTemplate>("rlm/rag-prompt");

const ragChain = RunnableSequence.from([
  {
    context: retriever.pipe(formatDocumentsAsString),
    question: new RunnablePassthrough(),
  },
  prompt,
  llm,
  new StringOutputParser(),
]);

for await (const chunk of await ragChain.stream(
  "What is task decomposition?",
)) {
  console.log(chunk);
}
