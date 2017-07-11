package spawnn.gui;

import java.util.EventListener;

public interface TrainingListener extends EventListener {
	public void trainingResultsAvailable(TrainingEvent te);
}
