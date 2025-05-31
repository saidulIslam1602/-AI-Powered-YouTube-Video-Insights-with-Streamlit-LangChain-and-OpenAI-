"""Multi-provider LLM system for advanced AI capabilities."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import asyncio

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, BaseMessage
from langchain.callbacks import StreamingStdOutCallbackHandler

from src.config.settings import settings
from src.utils.logger import logger
from src.utils.exceptions import ConfigurationError

class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"

@dataclass
class LLMConfig:
    """Configuration for LLM providers."""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2000
    streaming: bool = False
    custom_params: Dict[str, Any] = None

class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = None
        self._initialize_model()
    
    @abstractmethod
    def _initialize_model(self):
        """Initialize the specific LLM model."""
        pass
    
    @abstractmethod
    async def generate_response(self, messages: List[BaseMessage]) -> str:
        """Generate response from messages."""
        pass
    
    @abstractmethod
    async def generate_streaming_response(self, messages: List[BaseMessage]):
        """Generate streaming response from messages."""
        pass
    
    def create_messages(self, system_prompt: str, human_prompt: str) -> List[BaseMessage]:
        """Create message list from prompts."""
        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider."""
    
    def _initialize_model(self):
        """Initialize OpenAI model."""
        try:
            self.model = ChatOpenAI(
                model=self.config.model_name,
                openai_api_key=self.config.api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                streaming=self.config.streaming
            )
            logger.info(f"Initialized OpenAI model: {self.config.model_name}")
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize OpenAI model: {e}")
    
    async def generate_response(self, messages: List[BaseMessage]) -> str:
        """Generate response using OpenAI."""
        try:
            response = await asyncio.to_thread(self.model.invoke, messages)
            return response.content
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise
    
    async def generate_streaming_response(self, messages: List[BaseMessage]):
        """Generate streaming response using OpenAI."""
        try:
            async for chunk in self.model.astream(messages):
                yield chunk.content
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise

class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider."""
    
    def _initialize_model(self):
        """Initialize Anthropic model."""
        try:
            from langchain_anthropic import ChatAnthropic
            
            self.model = ChatAnthropic(
                model=self.config.model_name,
                anthropic_api_key=self.config.api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            logger.info(f"Initialized Anthropic model: {self.config.model_name}")
        except ImportError:
            raise ConfigurationError("Anthropic dependencies not installed. Run: pip install langchain-anthropic")
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Anthropic model: {e}")
    
    async def generate_response(self, messages: List[BaseMessage]) -> str:
        """Generate response using Anthropic."""
        try:
            response = await asyncio.to_thread(self.model.invoke, messages)
            return response.content
        except Exception as e:
            logger.error(f"Anthropic generation error: {e}")
            raise
    
    async def generate_streaming_response(self, messages: List[BaseMessage]):
        """Generate streaming response using Anthropic."""
        try:
            async for chunk in self.model.astream(messages):
                yield chunk.content
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise

class CohereProvider(BaseLLMProvider):
    """Cohere provider."""
    
    def _initialize_model(self):
        """Initialize Cohere model."""
        try:
            from langchain_cohere import ChatCohere
            
            self.model = ChatCohere(
                model=self.config.model_name,
                cohere_api_key=self.config.api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            logger.info(f"Initialized Cohere model: {self.config.model_name}")
        except ImportError:
            raise ConfigurationError("Cohere dependencies not installed. Run: pip install langchain-cohere")
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Cohere model: {e}")
    
    async def generate_response(self, messages: List[BaseMessage]) -> str:
        """Generate response using Cohere."""
        try:
            response = await asyncio.to_thread(self.model.invoke, messages)
            return response.content
        except Exception as e:
            logger.error(f"Cohere generation error: {e}")
            raise
    
    async def generate_streaming_response(self, messages: List[BaseMessage]):
        """Generate streaming response using Cohere."""
        try:
            async for chunk in self.model.astream(messages):
                yield chunk.content
        except Exception as e:
            logger.error(f"Cohere streaming error: {e}")
            raise

class HuggingFaceProvider(BaseLLMProvider):
    """Hugging Face provider for open-source models."""
    
    def _initialize_model(self):
        """Initialize Hugging Face model."""
        try:
            from langchain_community.llms import HuggingFacePipeline
            from transformers import pipeline
            
            # Create Hugging Face pipeline
            pipe = pipeline(
                "text-generation",
                model=self.config.model_name,
                max_length=self.config.max_tokens,
                temperature=self.config.temperature,
                device_map="auto" if self.config.custom_params.get("use_gpu", False) else None
            )
            
            self.model = HuggingFacePipeline(pipeline=pipe)
            logger.info(f"Initialized Hugging Face model: {self.config.model_name}")
        except ImportError:
            raise ConfigurationError("Hugging Face dependencies not installed. Run: pip install transformers torch")
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize Hugging Face model: {e}")
    
    async def generate_response(self, messages: List[BaseMessage]) -> str:
        """Generate response using Hugging Face."""
        try:
            # Convert messages to text format
            prompt = self._messages_to_prompt(messages)
            response = await asyncio.to_thread(self.model, prompt)
            return response
        except Exception as e:
            logger.error(f"Hugging Face generation error: {e}")
            raise
    
    async def generate_streaming_response(self, messages: List[BaseMessage]):
        """Generate streaming response using Hugging Face."""
        # Simplified implementation - HF doesn't natively support streaming
        response = await self.generate_response(messages)
        for chunk in response.split():
            yield chunk + " "
    
    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert LangChain messages to prompt format."""
        prompt_parts = []
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt_parts.append(f"System: {message.content}")
            elif isinstance(message, HumanMessage):
                prompt_parts.append(f"Human: {message.content}")
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)

class LocalLLMProvider(BaseLLMProvider):
    """Local LLM provider for running models locally."""
    
    def _initialize_model(self):
        """Initialize local model."""
        try:
            from langchain_community.llms import Ollama
            
            self.model = Ollama(
                model=self.config.model_name,
                base_url=self.config.base_url or "http://localhost:11434",
                temperature=self.config.temperature
            )
            logger.info(f"Initialized local model: {self.config.model_name}")
        except ImportError:
            raise ConfigurationError("Ollama dependencies not installed. Run: pip install ollama")
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize local model: {e}")
    
    async def generate_response(self, messages: List[BaseMessage]) -> str:
        """Generate response using local model."""
        try:
            prompt = self._messages_to_prompt(messages)
            response = await asyncio.to_thread(self.model, prompt)
            return response
        except Exception as e:
            logger.error(f"Local model generation error: {e}")
            raise
    
    async def generate_streaming_response(self, messages: List[BaseMessage]):
        """Generate streaming response using local model."""
        try:
            prompt = self._messages_to_prompt(messages)
            async for chunk in self.model.astream(prompt):
                yield chunk
        except Exception as e:
            logger.error(f"Local model streaming error: {e}")
            raise
    
    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert messages to prompt format."""
        prompt_parts = []
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt_parts.append(f"System: {message.content}")
            elif isinstance(message, HumanMessage):
                prompt_parts.append(f"Human: {message.content}")
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)

class LLMManager:
    """Manager for multiple LLM providers."""
    
    def __init__(self):
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.default_provider = None
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize configured providers."""
        # Primary provider (OpenAI by default)
        primary_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens
        )
        
        self.add_provider("primary", primary_config)
        self.default_provider = "primary"
        
        # Add additional providers if configured
        if hasattr(settings, 'anthropic_api_key') and settings.anthropic_api_key:
            anthropic_config = LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model_name=getattr(settings, 'anthropic_model', 'claude-3-sonnet-20240229'),
                api_key=settings.anthropic_api_key,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens
            )
            self.add_provider("anthropic", anthropic_config)
        
        # Add local provider if configured
        if hasattr(settings, 'local_model_enabled') and settings.local_model_enabled:
            local_config = LLMConfig(
                provider=LLMProvider.LOCAL,
                model_name=getattr(settings, 'local_model_name', 'llama2'),
                base_url=getattr(settings, 'local_model_url', 'http://localhost:11434'),
                temperature=settings.temperature,
                max_tokens=settings.max_tokens
            )
            self.add_provider("local", local_config)
    
    def add_provider(self, name: str, config: LLMConfig):
        """Add a new provider."""
        try:
            if config.provider == LLMProvider.OPENAI:
                provider = OpenAIProvider(config)
            elif config.provider == LLMProvider.ANTHROPIC:
                provider = AnthropicProvider(config)
            elif config.provider == LLMProvider.COHERE:
                provider = CohereProvider(config)
            elif config.provider == LLMProvider.HUGGINGFACE:
                provider = HuggingFaceProvider(config)
            elif config.provider == LLMProvider.LOCAL:
                provider = LocalLLMProvider(config)
            else:
                raise ConfigurationError(f"Unsupported provider: {config.provider}")
            
            self.providers[name] = provider
            logger.info(f"Added provider: {name} ({config.provider.value})")
            
        except Exception as e:
            logger.error(f"Failed to add provider {name}: {e}")
            raise
    
    def get_provider(self, name: Optional[str] = None) -> BaseLLMProvider:
        """Get provider by name or default."""
        provider_name = name or self.default_provider
        
        if provider_name not in self.providers:
            raise ConfigurationError(f"Provider {provider_name} not found")
        
        return self.providers[provider_name]
    
    def list_providers(self) -> List[str]:
        """List available providers."""
        return list(self.providers.keys())
    
    async def generate_with_fallback(
        self, 
        messages: List[BaseMessage], 
        preferred_provider: Optional[str] = None
    ) -> str:
        """Generate response with fallback to other providers."""
        providers_to_try = [preferred_provider] if preferred_provider else []
        providers_to_try.extend([name for name in self.providers.keys() if name != preferred_provider])
        
        last_error = None
        
        for provider_name in providers_to_try:
            if provider_name not in self.providers:
                continue
                
            try:
                provider = self.providers[provider_name]
                response = await provider.generate_response(messages)
                logger.info(f"Successfully generated response using {provider_name}")
                return response
                
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {e}")
                last_error = e
                continue
        
        raise Exception(f"All providers failed. Last error: {last_error}")
    
    async def compare_providers(
        self, 
        messages: List[BaseMessage], 
        providers: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """Compare responses from multiple providers."""
        providers_to_compare = providers or list(self.providers.keys())
        results = {}
        
        for provider_name in providers_to_compare:
            if provider_name not in self.providers:
                continue
                
            try:
                provider = self.providers[provider_name]
                response = await provider.generate_response(messages)
                results[provider_name] = response
            except Exception as e:
                logger.error(f"Provider {provider_name} failed: {e}")
                results[provider_name] = f"Error: {str(e)}"
        
        return results

# Global LLM manager instance
llm_manager = LLMManager() 