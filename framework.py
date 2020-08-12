from parlai.core.worlds import World
from parlai.chat_service.services.messenger.worlds import OnboardWorld
from parlai.core.agents import create_agent_from_shared

# Create a human agent
def _create_agent(self, task_id, socketID):
        """
        Initialize an agent and return it.
        Called each time an agent is placed into a new task.
        :param task_id:
            string task identifier
        :param agent_id:
            int agent id
        """
        return WebsocketAgent(self.opt, self, socketID, task_id)

# Create a model
def _load_model(self):
        """
        Load model if necessary.
        """
        if 'models' in self.opt and self.should_load_model:
            model_params = {}
            model_info = {}
            for model in self.opt['models']:
                model_opt = self.opt['models'][model]
                override = model_opt.get('override', {})
                if type(override) is list:
                    model_opt['override'] = override[0]
                model_params[model] = create_agent(model_opt).share()
                model_info[model] = {'override': override}
            self.runner_opt['model_info'] = model_info
            self.runner_opt['shared_bot_params'] = model_params
# Import and subclass
class MessengerBotChatTaskWorld(World):
    """
    Example one person world that talks to a provided agent (bot).
    """

    MAX_AGENTS = 1
    MODEL_KEY = 'blender_90M'

    def __init__(self, opt, agent, bot):
        self.agent = agent
        self.episodeDone = False
        self.model = bot
        self.first_time = True

    @staticmethod
    def generate_world(opt, agents):
        if opt['models'] is None:
            raise RuntimeError("Model must be specified")
        return MessengerBotChatTaskWorld(
            opt,
            agents[0],
            create_agent_from_shared(
                opt['shared_bot_params'][MessengerBotChatTaskWorld.MODEL_KEY]
            ),
        )

    @staticmethod
    def assign_roles(agents):
        agents[0].disp_id = 'ChatbotAgent'

    def parley(self):
        if self.first_time:
            self.agent.observe(
                {
                    'id': 'World',
                    'text': 'Welcome to the ParlAI Chatbot demo. '
                    'You are now paired with a bot - feel free to send a message.'
                    'Type [DONE] to finish the chat.',
                }
            )
            self.first_time = False
        a = self.agent.act()
        if a is not None:
            if '[DONE]' in a['text']:
                self.episodeDone = True
            else:
                print("===act====")
                print(a)
                print("~~~~~~~~~~~")
                self.model.observe(a)
                response = self.model.act()
                print("===response====")
                print(response)
                print("~~~~~~~~~~~")
                self.agent.observe(response)

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        self.agent.shutdown()
        
