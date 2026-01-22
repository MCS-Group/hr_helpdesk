from typing import List
from domain.entities import MCSDocumentChunk, QueryResult
from infrastructure.interfaces import ILLMService, IKnowledgeBase
from llama_index.core.prompts import RichPromptTemplate
from llama_index.llms.openai import OpenAI

class LLMService(ILLMService):
    """
    Service layer for synthesizing responses via LLM based on provided context and prompts configured.
    """

    def __init__(self, prompt_template: str | None = None) -> None:
        if prompt_template is None:
            
            prompt_template = """
You are a helpful assistant that provides information about HR policies based on the chunks of context provided further down in Mongolian language.

You are asked to answer the following question:\n{{question}}
Read the following context before answering:\n{{context}}

If appropriate, you have to structure your output using bullet points or numbered lists for clarity. If necessary try not to be verbose, but do not leave out important details or lose information.

Always respond in Mongolian language while making your reasoning in English.
"""

        self._prompt_template = prompt_template
        self._llm = OpenAI(
            model="chatgpt-4o-latest",
            temperature=0.2
        )
    
    async def synthesize_response(self, user_question: str, context: List[MCSDocumentChunk]) -> str:
        """
        Generate a response from the LLM based on the given context.

        :param context: List of MCSDocumentChunk providing context for the response.
        :return: Generated response as a string.
        """
        prompt = RichPromptTemplate(self._prompt_template)
        
        context_texts = [chunk.content for chunk in context]
        combined_context = "\n".join(context_texts)

        filled_prompt = prompt.format_messages(
            question=user_question,
            context=combined_context
        )

        response = await self._llm.achat(filled_prompt)
        return response.message.content
        