package spawnn.gui;

import javax.swing.JPanel;
import javax.swing.JLabel;
import net.miginfocom.swing.MigLayout;
import javax.swing.JTextField;

public class CNGPanel extends JPanel {

	private static final long serialVersionUID = -4550923008702082066L;
	private JTextField textField;

	public CNGPanel() {
		setLayout(new MigLayout());
				
		JLabel lblSNS = new JLabel("l:");
		lblSNS.setToolTipText("Neighborhood size");
		add(lblSNS, "");
		
		textField = new JTextField();
		textField.setText("1");
		add(textField, "growx");
		textField.setColumns(10);

	}
	
	public int getSNS() {
		return Integer.parseInt(textField.getText() );
	}

}
