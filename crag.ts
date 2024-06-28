import { StringOutputParser } from "@langchain/core/output_parsers";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import {
  Runnable,
  RunnableConfig,
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import { ChatGroq } from "@langchain/groq";
import { NomicEmbeddings } from "@langchain/nomic";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { formatDocumentsAsString } from "langchain/util/document";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

const loader = new CheerioWebBaseLoader(
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
);

const docs = await loader.load();

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

const splits = await textSplitter.invoke(docs);

const vecStore = await MemoryVectorStore.fromDocuments(
  splits,
  new NomicEmbeddings(),
);

const retriever = vecStore.asRetriever();

const llm = new ChatGroq({
  model: "mixtral-8x7b-32768",
  temperature: 0,
});

const contextualizeQSystemPrompt = `Given a chat history and the latest user question
which might reference context in the chat history, formulate a standalone question
which can be understood without the chat history. Do NOT answer the question,
just reformulate it if needed and otherwise return it as is.`;

const contextualizeQPrompt = ChatPromptTemplate.fromMessages([
  ["system", contextualizeQSystemPrompt],
  new MessagesPlaceholder("chat_history"),
  ["human", "{question}"],
]);

const contextualizeQChain = contextualizeQPrompt
  .pipe(llm)
  .pipe(new StringOutputParser());

const qaSystemPrompt = `You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

{context}`;

const qaPrompt = ChatPromptTemplate.fromMessages([
  ["system", qaSystemPrompt],
  new MessagesPlaceholder("chat_history"),
  ["human", "{question}"],
]);

const contextualizeQuestion = (input: Record<string, unknown>) => {
  if ("chat_history" in input) return contextualizeQChain;
  return input.question;
};

const ragChain = RunnableSequence.from([
  RunnablePassthrough.assign({
    context: (input: Record<string, unknown>) => {
      if ("chat_history" in input) {
        const chain = contextualizeQuestion(input) as Runnable<
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          any,
          string,
          RunnableConfig
        >;
        return chain.pipe(retriever).pipe(formatDocumentsAsString);
      }
      return "";
    },
  }),
  qaPrompt,
  llm,
]);

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let chat_history: any[] = [];

const question = "What is task decomposition?";
const aiMsg = await ragChain.invoke({ question, chat_history });
console.log(aiMsg);
chat_history = chat_history.concat(aiMsg);

const secondQuestion = "What are common ways of doing it?";
await ragChain.invoke({ question: secondQuestion, chat_history });
