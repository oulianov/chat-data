"""Create a ConversationalRetrievalChain for question/answering."""
import numpy as np
from typing import Dict, Any, Optional

from langchain.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForChainRun,
)
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import _get_chat_history
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.base import VectorStore
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings

# TODO : Use summary of summaries in the prompt template to give context
# TODO : Understand if the user is talking about a document of the collection, and if so which one


class QuestionDatabase:
    # Idea : use this as a way to check if the question is close
    # to something we know. If so, serve the reply.

    def __init__(self, embeddings: OpenAIEmbeddings):
        self.embeddings = embeddings  # embedding function
        self.stored_queries: Dict[str, np.ndarray] = {}
        self.stored_reponses: Dict[str, str] = {}

    def get(self, query: str, min_similarity: float = 0.9) -> Optional[str]:
        query_emb = np.array(self.embeddings.embed_query(query))
        closest_stored_query = None
        closest_sim = None
        for stored_query, stored_vector in self.stored_queries.items():
            similarity = (stored_vector @ query_emb) / (
                np.linalg.norm(stored_vector) * np.linalg.norm(query_emb)
            )
            print(similarity)
            if not closest_stored_query or not closest_sim:
                if similarity > min_similarity:
                    closest_stored_query = stored_query
                    closest_sim = similarity
            else:
                if similarity > closest_sim:
                    closest_stored_query = stored_query
                    closest_sim = similarity
        if closest_stored_query:
            return self.stored_reponses[closest_stored_query]
        else:
            return None

    def update(self, query: str, response: str):
        query_emb = np.array(self.embeddings.embed_query(query))
        self.stored_queries[query] = query_emb
        self.stored_reponses[query] = response


class CustomConversationalRetrievalChain(ConversationalRetrievalChain):
    question_database: QuestionDatabase

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        output: Dict[str, Any] = {}

        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        question = inputs["question"]
        get_chat_history = self.get_chat_history or _get_chat_history
        chat_history_str = get_chat_history(inputs["chat_history"])
        if chat_history_str:
            callbacks = _run_manager.get_child()
            new_question = await self.question_generator.arun(
                question=question, chat_history=chat_history_str, callbacks=callbacks
            )
        else:
            new_question = question

        # verify if the question is in the question database
        print("calling db")
        answer_from_db = self.question_database.get(new_question)
        print(answer_from_db)
        if answer_from_db:
            answer = answer_from_db
            await _run_manager.handlers[0].on_rep(message=answer)
            output["message"] = answer
        else:
            docs = await self._aget_docs(new_question, inputs)
            new_inputs = inputs.copy()
            new_inputs["question"] = new_question
            new_inputs["chat_history"] = chat_history_str
            answer = await self.combine_docs_chain.arun(
                input_documents=docs, callbacks=_run_manager.get_child(), **new_inputs
            )
            if self.return_source_documents:
                output["source_documents"] = docs
        output[self.output_key] = answer
        if self.return_generated_question:
            output["generated_question"] = new_question
        return output


CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    template="""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:""",
)

QA_PROMPT = PromptTemplate.from_template(
    template="""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:""",
)


def get_chain(
    vectorstore: VectorStore,
    question_handler,
    insta_rep_handler,
    stream_handler,
    tracing: bool = False,
) -> ConversationalRetrievalChain:
    """Create a ConversationalRetrievalChain for question/answering."""
    # Construct a ConversationalRetrievalChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([insta_rep_handler])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()  # type: ignore
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)

    question_gen_llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        verbose=True,
        callback_manager=question_manager,
    )  # type: ignore
    streaming_llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0,
    )  # type: ignore
    embeddings = OpenAIEmbeddings()
    question_database = QuestionDatabase(embeddings=embeddings)
    question_database.update("who made this app ?", "It was made by Nicolas.")
    question_database.update(
        "who is the author ?", "I'm afraid I can't tell you that, Dave."
    )

    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
    )
    doc_chain = load_qa_chain(
        streaming_llm, chain_type="stuff", prompt=QA_PROMPT, callback_manager=manager
    )

    qa = CustomConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(k=3),
        combine_docs_chain=doc_chain,
        question_generator=question_generator,
        callback_manager=manager,
        return_source_documents=False,
        question_database=question_database,
    )
    return qa
