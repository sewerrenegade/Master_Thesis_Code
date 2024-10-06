from pytorch_lightning.callbacks import ModelCheckpoint

class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        self.save_top_k = kwargs.get("top_k_to_save", 3)
        if "top_k_to_save" in kwargs:
            del kwargs["top_k_to_save"] 
        super().__init__(*args, **kwargs) # avoid native checkpoint saving methods, only need inherited behavior
        
        assert isinstance(self.save_top_k,int)
        self.best_models = []  # A list to store the top 3 models
    def get_best_path(self):
        return self.best_models[0][2]
    def _compare_metrics(self, current_metrics,trainer, pl_module):
        """
        Compare the current metrics (val_correct_epoch, val_loss_epoch) with the saved models.
        The goal is to keep the top 3 models, prioritize val_correct_epoch and resolve ties using val_loss_epoch.
        """
        current_correct = current_metrics["val_correct_epoch"]
        current_loss = current_metrics["val_loss_epoch"]

        # Insert current metrics into the best models list based on conditions
        inserted = False
        for i, (correct, loss, _) in enumerate(self.best_models):
            if current_correct > correct or (current_correct == correct and current_loss < loss):
                self.best_models.insert(i, (current_correct, current_loss,  self._save_model(trainer, pl_module)))  # insert in the correct position
                inserted = True
                break
        
        if not inserted and len(self.best_models) < self.save_top_k:
            self.best_models.append((current_correct, current_loss, None))

        # Ensure we only keep the top 3 models
        if len(self.best_models) > self.save_top_k:
            self.best_models.pop()

    def on_validation_end(self, trainer, pl_module):
        """
        Called at the end of validation. This is where you determine whether to save the model based on the metrics.
        """
        # Get the current metrics
        logs = trainer.callback_metrics
        current_metrics = {
            "val_correct_epoch": logs.get("val_correct_epoch").item(),
            "val_loss_epoch": logs.get("val_loss_epoch").item()
        }
        # Compare and update the best models
        if pl_module.current_epoch != 0:
            self._compare_metrics(current_metrics,trainer,pl_module)

            # If the model should be saved, use the usual save method
            if any(x is None for x in self.best_models[-1]):
                # Update the model path for the latest saved model
                self.best_models[-1] = (self.best_models[-1][0], self.best_models[-1][1], self._save_model(trainer, pl_module))
            self.custom_best_model_path = self.best_models[0][-1]
            # Print best models for debugging purposes
            print(f"Best models (sorted): {self.best_models}")

    def _save_model(self, trainer, pl_module):
        """
        Use the original ModelCheckpoint's save logic to save the model.
        """
        name_metrics = {"epoch":trainer.current_epoch}
        name_metrics.update( trainer.callback_metrics)
        filepath = self.format_checkpoint_name(name_metrics)
        self._save_checkpoint(trainer, filepath)
        return filepath


# checkpoint_callback = CustomModelCheckpoint(
#     save_weights_only=True,
#     filename="{epoch}-{val_correct_epoch:.4f}-{val_loss_epoch:.4f}",
#     verbose=True,
#     save_top_k=3
# )