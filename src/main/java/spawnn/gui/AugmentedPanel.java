package spawnn.gui;

import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

import net.miginfocom.swing.MigLayout;

public class AugmentedPanel extends JPanel {

	private static final long serialVersionUID = -771653004644258074L;
	private JTextField textField;

	public AugmentedPanel() {
		setLayout(new MigLayout());
				
		JLabel lblAlpha = new JLabel("Alpha:");
		lblAlpha.setToolTipText("Weight of spatial coordinates");
		add(lblAlpha, "");
		
		textField = new JTextField();
		textField.setText("0.5");
		add(textField, "growx");
		textField.setColumns(10);

	}
	
	public double getAlpha() {
		return Double.parseDouble(textField.getText());
	}

}
