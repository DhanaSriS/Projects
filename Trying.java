import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import java.io.File;
import java.io.IOException;
import java.sql.SQLException;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVPrinter;
import java.io.*;
import java.sql.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Trying {

    private static final Logger logger = LoggerFactory.getLogger(Trying.class);

    public static void main(String[] args) {
        try (PDDocument document = PDDocument.load(new File("C:\\Users\\dhana\\Downloads\\Anjana bill new.pdf"))) {
            PDFTextStripper stripper = new PDFTextStripper();
            String content = stripper.getText(document);
            String[] lines = content.split("\\r?\\n"); // Split text into lines
            String[][] columns = new String[lines.length][]; // Array to hold columns for each line
            for (int i = 0; i < lines.length; i++) {
                columns[i] = splitIntoColumns(lines[i]); // Split line into columns
            }
            for (int i = 0; i < lines.length; i++) {
                System.out.println(columns[i][0]);
            }

            String line = columns[16][0];
            String invoice = columns[19][0];
            String[] parts = line.split("\\s{2,}");

            String productName = parts[0];
            String quantity = parts[1];
            String price = parts[2];

            String cus_name = columns[15][0];
            String mode = columns[20][0];
            System.out.println("Invoice: " + invoice);
            System.out.println("Product Name: " + productName);
            System.out.println("Quantity: " + quantity);
            System.out.println("Price: " + price);
            System.out.println("Customer name: " + cus_name);
            System.out.println("Mode of payment: " + mode);

            Connection con = DriverManager.getConnection("jdbc:mysql://localhost:3306/Invoice", "root", "Dhana@21");
            boolean invoiceExists = checkInvoiceExists(con, invoice);
            if (invoiceExists) {
                saveToDatabase(invoice, productName, quantity, price, cus_name, mode);
                saveToCSV(invoice, productName, quantity, price, cus_name, mode);
            } else {
                System.out.println("Invoice does not exist in the database.");
            }

        } catch (SQLException e) {
            logger.error("SQLException occurred", e);
        } catch (IOException e) {
            logger.error("IOException occurred", e);
        }

    }

    private static String[] splitIntoColumns(String line) {
        return new String[] { line };
    }

    private static void saveToDatabase(String invoice, String productName, String quantity, String price,
            String cus_name, String mode) {
        String query = "INSERT INTO bills (invoice, productName, quantity, price, cusName, paymode) VALUES (?, ?, ?, ?, ?, ?)";

        try (Connection con = DriverManager.getConnection("jdbc:mysql://localhost:3306/Bill", "root", "Dhana@21");
                PreparedStatement pst = con.prepareStatement(query)) {
            pst.setString(1, invoice);
            pst.setString(2, productName);
            pst.setString(3, quantity);
            pst.setString(4, price);
            pst.setString(5, cus_name);
            pst.setString(6, mode);

            pst.executeUpdate();

            System.out.println("Data inserted successfully!");

        } catch (SQLException ex) {
            System.out.println(ex.getMessage());
        }
    }

    private static void saveToCSV(String invoice, String productName, String quantity, String price, String cusName,
            String mode) {
        // CSV file path
        try (Writer writer = new FileWriter("D:\\C_files\\Java\\Bill parser\\bill.csv");
                CSVPrinter csvPrinter = new CSVPrinter(writer, CSVFormat.DEFAULT)) {
            csvPrinter.printRecord("Invoice", "Product Name", "Quantity", "Price", "Customer Name", "Mode of Payment");

            csvPrinter.printRecord(invoice, productName, quantity, price, cusName, mode);

            System.out.println("Data saved to CSV file successfully!");

        } catch (IOException e) {
            logger.error("IOException occurred", e);
        }
    }

    private static boolean checkInvoiceExists(Connection con, String invoiceToSearch) throws SQLException {
        String query = "SELECT * FROM inv WHERE invoice = ?";
        PreparedStatement pst = con.prepareStatement(query);
        pst.setString(1, invoiceToSearch);
        ResultSet rs = pst.executeQuery();
        return rs.next();
    }

}
