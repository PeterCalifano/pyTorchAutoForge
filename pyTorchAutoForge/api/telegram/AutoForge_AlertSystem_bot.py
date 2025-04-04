# Telegram API bot address: https://api.telegram.org/bot7475996719:AAHos6jPYrZWNNrzg7CpGWNSxmF2z7n6_rc/getMe
# API address for updates: https://api.telegram.org/bot7475996719:AAHos6jPYrZWNNrzg7CpGWNSxmF2z7n6_rc/getUpdates
# PeterC chat_id: 293510580
from exceptiongroup import catch
from telegram import Bot
import json

try:
    def GetDefaultBotToken() -> str:
        with open('bot_token_private.json') as file:
            data = json.load(file)
            return data['token']

    def GetDefaultChatID() -> str:
        with open('bot_token_private.json') as file:
            data = json.load(file)
            return data['chat_id']

    class AutoForgeAlertSystemBot(Bot):

        def __init__(self, token: str = GetDefaultBotToken(), chat_id: str = GetDefaultChatID()) -> "AutoForgeAlertSystemBot":
            if token is None:
                raise ValueError("Token is required to create a bot instance")
            super().__init__(token=token)

            self.chat_id_ = chat_id # DEVNOTE: not working, but not sure why?

        
        def sendMessage(self, text_string: str) -> bool:
            if self.chat_id is None:
                Warning("Chat ID is not set. No messahge has been sent")
                return False
            else:
                self.send_message(chat_id=self.chat_id, text=text_string)
                return True
except:
    print('Failed to import functions in telegram api module.')

if __name__ == "__main__":
    # Replace with your bot's token and chat ID
    with open('bot_token_private.json') as file:
        data = json.load(file)
        token = data['token']
        chat_id = data['chat_id']

    text = 'Hello, this is a test message from pytorch auto-forge alert system!'
    bot = AutoForgeAlertSystemBot(token=token, chat_id=chat_id)
    bot.sendMessage(text)
