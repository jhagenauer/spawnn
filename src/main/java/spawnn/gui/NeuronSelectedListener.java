package spawnn.gui;

import java.util.EventListener;

public interface NeuronSelectedListener <T> extends EventListener {
	public void neuronSelectedOccured(NeuronSelectedEvent<T> evt);
}
