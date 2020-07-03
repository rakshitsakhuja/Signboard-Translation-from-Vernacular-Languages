from checkpoint import save_ckp
from config import parameters
from utils import Averager


def train_fn(start_epochs, epochs, train_loader, val_loader, model, device, optimizer, best_loss,checkpoint_path, best_model_path, lr_scheduler=None):
    print("Starting Training")
    model.train()

    loss_hist = Averager()
    itr = 1
    train_loss = []
    validation_loss = []
    for epoch in range(start_epochs, epochs + 1):
        loss_hist.reset()

        for images, targets, image_ids in train_loader:

            images = list(image.to(device) for image in images)
            # images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            
            loss_hist.send(loss_value)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if itr % 10 == 0:
                print(f"Iteration #{itr} loss: {loss_value}")

            itr += 1

        # update the learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        # val_loss = validate(val_loader, model, device)
        print(f"Epoch #{epoch} Train loss: {loss_hist.value}, Validation Loss : Commented")
        train_loss.append(loss_hist.value)
        # validation_loss.append(val_loss)
        checkpoint = {
            'epoch': epoch + 1,
            # 'best_loss': val_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        # if best_loss <= val_loss:
        #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(best_loss, val_loss))
        #     save_ckp(checkpoint, True, checkpoint_path, best_model_path)
        #     best_loss = val_loss

    return model, train_loss#, validation_loss


def validate(val_loader, model, device):
    model.eval()
    itr = 1
    loss_hist = Averager()
    loss_hist.reset()
    for images, targets, image_ids in val_loader:
      images = list(image.to(device) for image in images)
      targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
      loss_dict = model(images, targets)
      losses = sum(loss for loss in loss_dict)
      loss_value = losses.item()
      loss_hist.send(loss_value)
      if itr % 20 == 0:
        print(f"Iteration: {itr} loss: {loss_hist.value}")
      itr += 1
    return loss_hist.value
