import re
import ast
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, StoppingCriteria


# classes for SLM generation
class ActionBlockStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.reset("")

    def reset(self, prompt):
        self.prompt = prompt
        self.prompt_len = len(
            self.tokenizer.decode(self.tokenizer(prompt, return_tensors="pt").input_ids[0], skip_special_tokens=True)
        )

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        new_text = text[self.prompt_len :]

        idx = new_text.find("Action:")
        if idx == -1:
            return False
        new_text = new_text[idx:]
        brace_count = 0
        for c in new_text:
            if c == "{":
                brace_count += 1
            elif c == "}":
                brace_count -= 1
                if brace_count == 0:
                    return True
        return False


class SLM:
  """
  Class for importing and running an SLM.
  """
  def __init__(self, model_name):
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    self.stopping_criteria = ActionBlockStoppingCriteria(self.tokenizer)

  def run(self, messages, step_results=None, stopping_citeria=None):
    text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if stopping_citeria:
        self.stopping_criteria.reset(text)
    model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
    input_token_count = len(model_inputs.input_ids[0])
    if step_results:
        step_results["token_count"] = input_token_count

    if stopping_citeria:
      generated_ids = self.model.generate(
          **model_inputs,
          max_new_tokens=500,
          stopping_criteria=StoppingCriteriaList([self.stopping_criteria]),
          do_sample=True,
          temperature=0.7,
      )
    else:
      generated_ids = self.model.generate(
          **model_inputs,
          max_new_tokens=500,
          do_sample=True,
          temperature=0.7,
      )
    generated_ids = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output

# useful methods
def parse(output):
    """JSON extraction from LLM output."""
    output = output.replace("```json", "")
    output = output.replace("```", "")
    matches = re.findall(r"Action:\s*(\{.*\})", output, re.DOTALL)
    result = matches[-1]
    json_result = ast.literal_eval(result)
    return json_result
