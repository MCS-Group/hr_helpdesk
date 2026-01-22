from botbuilder.core import ActivityHandler, TurnContext
from botbuilder.schema import Activity, ActivityTypes
from botbuilder.core.teams import TeamsInfo

from services.knowledge_service import KnowledgeService
from services.llm_service import LLMService


from services.react_service import workflow

#kb_service = KnowledgeService(None)
#llm_service = LLMService(None)

class MCSHumanResourcesBot(ActivityHandler):
    
    async def on_message_activity(self, turn_context: TurnContext):
        
        await turn_context.send_activity(
            Activity(type=ActivityTypes.typing)
        )

        query = turn_context.activity.text.strip()

        member = await TeamsInfo.get_member(
            turn_context,
            turn_context.activity.from_property.id
        )

        email = member.email
        employee_query = f"Ажилтны email: {email}\nАжилтны асуулт: {query}"
        response = await workflow.run(user_msg=employee_query)
        
        #context = await kb_service.query(query, top_k=5)
        #response = await llm_service.synthesize_response(query, context.chunks)

        await turn_context.send_activity(
            Activity(
                type=ActivityTypes.message,
                text=response
            )
        )