package spawnn.gui;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;

import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

import net.miginfocom.swing.MigLayout;

public class WMCPanel extends JPanel implements ActionListener {

	private static final long serialVersionUID = 1671229973714048612L;
	private JTextField textField;
	private JTextField textField_1;
	private JButton btnDistMatrix;
	
	private JTextField textField_2;
	private JTextField textField_3;
	private JTextField selFile;
	
	private File distMapFile;
	

	public WMCPanel() {
		setLayout(new MigLayout());
		
		JLabel lblAlpha = new JLabel("Alpha:");
		add(lblAlpha, "");
		
		textField = new JTextField();
		textField.setText("0.5");
		textField.setColumns(10);
		add(textField, "pushx");
				
		JLabel lblBeta = new JLabel("Beta:");
		add(lblBeta, "");
		
		textField_1 = new JTextField();
		textField_1.setText("0.5");
		textField_1.setColumns(10);
		add(textField_1, "wrap");
				
		add( new JLabel("Dist. matrix:"),""); 
		
		selFile = new JTextField();
		selFile.setColumns(20);
		selFile.setEditable(true);
		add(selFile,"growx");
		
		btnDistMatrix = new JButton("Select...");
		btnDistMatrix.addActionListener(this);
		add(btnDistMatrix, "");
		
		
	}

	@Override
	public void actionPerformed(ActionEvent ae) {
		if( ae.getSource() == btnDistMatrix ) {
			JFileChooser fc = new JFileChooser();
			int state = fc.showOpenDialog(this);
			if( state == JFileChooser.APPROVE_OPTION ) { 
			      distMapFile = fc.getSelectedFile();
			      selFile.setText(""+distMapFile);
			}
		}	
	}
	
	public double getAlpha() {
		return Double.parseDouble(textField.getText());
	}
	
	public double getBeta() {
		return Double.parseDouble(textField_1.getText());
	}
			
	public File getDistMapFile() {
		return distMapFile;
	}
}
