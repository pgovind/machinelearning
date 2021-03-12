﻿//------------------------------------------------------------------------------
// <auto-generated>
//     This code was generated by a tool.
//     Runtime Version:4.0.30319.42000
//
//     Changes to this file may cause incorrect behavior and will be lost if
//     the code is regenerated.
// </auto-generated>
//------------------------------------------------------------------------------

namespace Microsoft.Data {
    using System;
    
    
    /// <summary>
    ///   A strongly-typed resource class, for looking up localized strings, etc.
    /// </summary>
    // This class was auto-generated by the StronglyTypedResourceBuilder
    // class via a tool like ResGen or Visual Studio.
    // To add or remove a member, edit your .ResX file then rerun ResGen
    // with the /str option, or rebuild your VS project.
    [global::System.CodeDom.Compiler.GeneratedCodeAttribute("System.Resources.Tools.StronglyTypedResourceBuilder", "16.0.0.0")]
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute()]
    [global::System.Runtime.CompilerServices.CompilerGeneratedAttribute()]
    internal class Strings {
        
        private static global::System.Resources.ResourceManager resourceMan;
        
        private static global::System.Globalization.CultureInfo resourceCulture;
        
        [global::System.Diagnostics.CodeAnalysis.SuppressMessageAttribute("Microsoft.Performance", "CA1811:AvoidUncalledPrivateCode")]
        internal Strings() {
        }
        
        /// <summary>
        ///   Returns the cached ResourceManager instance used by this class.
        /// </summary>
        [global::System.ComponentModel.EditorBrowsableAttribute(global::System.ComponentModel.EditorBrowsableState.Advanced)]
        internal static global::System.Resources.ResourceManager ResourceManager {
            get {
                if (object.ReferenceEquals(resourceMan, null)) {
                    global::System.Resources.ResourceManager temp = new global::System.Resources.ResourceManager("Microsoft.Data.Analysis.Tests.Strings", typeof(Strings).Assembly);
                    resourceMan = temp;
                }
                return resourceMan;
            }
        }
        
        /// <summary>
        ///   Overrides the current thread's CurrentUICulture property for all
        ///   resource lookups using this strongly typed resource class.
        /// </summary>
        [global::System.ComponentModel.EditorBrowsableAttribute(global::System.ComponentModel.EditorBrowsableState.Advanced)]
        internal static global::System.Globalization.CultureInfo Culture {
            get {
                return resourceCulture;
            }
            set {
                resourceCulture = value;
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Cannot cast column holding {0} values to type {1}.
        /// </summary>
        internal static string BadColumnCast {
            get {
                return ResourceManager.GetString("BadColumnCast", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Line {0} cannot be parsed with the current Delimiters..
        /// </summary>
        internal static string CannotParseWithDelimiters {
            get {
                return ResourceManager.GetString("CannotParseWithDelimiters", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Line {0} cannot be parsed with the current FieldWidths..
        /// </summary>
        internal static string CannotParseWithFieldWidths {
            get {
                return ResourceManager.GetString("CannotParseWithFieldWidths", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Cannot resize down.
        /// </summary>
        internal static string CannotResizeDown {
            get {
                return ResourceManager.GetString("CannotResizeDown", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Index cannot be greater than the Column&apos;s Length.
        /// </summary>
        internal static string ColumnIndexOutOfRange {
            get {
                return ResourceManager.GetString("ColumnIndexOutOfRange", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Comment token cannot contain whitespace.
        /// </summary>
        internal static string CommentTokenCannotContainWhitespace {
            get {
                return ResourceManager.GetString("CommentTokenCannotContainWhitespace", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to DataType.
        /// </summary>
        internal static string DataType {
            get {
                return ResourceManager.GetString("DataType", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Delimiter cannot be new line characters.
        /// </summary>
        internal static string DelimiterCannotBeNewlineChar {
            get {
                return ResourceManager.GetString("DelimiterCannotBeNewlineChar", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Length (excluding null values).
        /// </summary>
        internal static string DescriptionMethodLength {
            get {
                return ResourceManager.GetString("DescriptionMethodLength", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to DataFrame already contains a column called {0}.
        /// </summary>
        internal static string DuplicateColumnName {
            get {
                return ResourceManager.GetString("DuplicateColumnName", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Delimiters is empty..
        /// </summary>
        internal static string EmptyDelimiters {
            get {
                return ResourceManager.GetString("EmptyDelimiters", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to FieldWidths is empty..
        /// </summary>
        internal static string EmptyFieldWidths {
            get {
                return ResourceManager.GetString("EmptyFieldWidths", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Empty file.
        /// </summary>
        internal static string EmptyFile {
            get {
                return ResourceManager.GetString("EmptyFile", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Exceeded maximum buffer size..
        /// </summary>
        internal static string ExceededMaxBufferSize {
            get {
                return ResourceManager.GetString("ExceededMaxBufferSize", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Parameter.Count exceeds the number of columns({0}) in the DataFrame .
        /// </summary>
        internal static string ExceedsNumberOfColumns {
            get {
                return ResourceManager.GetString("ExceedsNumberOfColumns", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Parameter.Count exceeds the number of rows({0}) in the DataFrame .
        /// </summary>
        internal static string ExceedsNumberOfRows {
            get {
                return ResourceManager.GetString("ExceedsNumberOfRows", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Expected either {0} or {1} to be provided.
        /// </summary>
        internal static string ExpectedEitherGuessRowsOrDataTypes {
            get {
                return ResourceManager.GetString("ExpectedEitherGuessRowsOrDataTypes", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to {0} not found..
        /// </summary>
        internal static string FileNotFound {
            get {
                return ResourceManager.GetString("FileNotFound", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to A double quote is not a legal delimiter when HasFieldsEnclosedInQuotes is set to True..
        /// </summary>
        internal static string IllegalQuoteDelimiter {
            get {
                return ResourceManager.GetString("IllegalQuoteDelimiter", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Column is immutable.
        /// </summary>
        internal static string ImmutableColumn {
            get {
                return ResourceManager.GetString("ImmutableColumn", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Inconsistent null bitmap and data buffer lengths.
        /// </summary>
        internal static string InconsistentNullBitMapAndLength {
            get {
                return ResourceManager.GetString("InconsistentNullBitMapAndLength", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Inconsistent null bitmaps and NullCounts.
        /// </summary>
        internal static string InconsistentNullBitMapAndNullCount {
            get {
                return ResourceManager.GetString("InconsistentNullBitMapAndNullCount", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Column does not exist.
        /// </summary>
        internal static string InvalidColumnName {
            get {
                return ResourceManager.GetString("InvalidColumnName", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to All field widths, except the last element, must be greater than zero. A field width less than or equal to zero in the last element indicates the last field is of variable length..
        /// </summary>
        internal static string InvalidFieldWidths {
            get {
                return ResourceManager.GetString("InvalidFieldWidths", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Line {0} has less columns than expected.
        /// </summary>
        internal static string LessColumnsThatExpected {
            get {
                return ResourceManager.GetString("LessColumnsThatExpected", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Line {0} cannot be read because it exceeds the max line size..
        /// </summary>
        internal static string LineExceedsMaxLineSize {
            get {
                return ResourceManager.GetString("LineExceedsMaxLineSize", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to MapIndices exceeds column length.
        /// </summary>
        internal static string MapIndicesExceedsColumnLenth {
            get {
                return ResourceManager.GetString("MapIndicesExceedsColumnLenth", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Column lengths are mismatched.
        /// </summary>
        internal static string MismatchedColumnLengths {
            get {
                return ResourceManager.GetString("MismatchedColumnLengths", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Expected column to hold values of type {0}.
        /// </summary>
        internal static string MismatchedColumnValueType {
            get {
                return ResourceManager.GetString("MismatchedColumnValueType", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to rowCount differs from Column length for Column .
        /// </summary>
        internal static string MismatchedRowCount {
            get {
                return ResourceManager.GetString("MismatchedRowCount", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Expected value to be of type {0}.
        /// </summary>
        internal static string MismatchedValueType {
            get {
                return ResourceManager.GetString("MismatchedValueType", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Expected value to be of type {0}, {1} or {2}.
        /// </summary>
        internal static string MultipleMismatchedValueType {
            get {
                return ResourceManager.GetString("MultipleMismatchedValueType", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Expected a seekable stream.
        /// </summary>
        internal static string NonSeekableStream {
            get {
                return ResourceManager.GetString("NonSeekableStream", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Delimiters is null..
        /// </summary>
        internal static string NullDelimiters {
            get {
                return ResourceManager.GetString("NullDelimiters", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to FieldWidths is null..
        /// </summary>
        internal static string NullFieldWidths {
            get {
                return ResourceManager.GetString("NullFieldWidths", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to numeric column.
        /// </summary>
        internal static string NumericColumnType {
            get {
                return ResourceManager.GetString("NumericColumnType", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to {0} must be greater than 0.
        /// </summary>
        internal static string PositiveNumberOfCharacters {
            get {
                return ResourceManager.GetString("PositiveNumberOfCharacters", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Cannot span multiple buffers.
        /// </summary>
        internal static string SpansMultipleBuffers {
            get {
                return ResourceManager.GetString("SpansMultipleBuffers", resourceCulture);
            }
        }
        
        /// <summary>
        ///   Looks up a localized string similar to Stream doesn&apos;t support reading.
        /// </summary>
        internal static string StreamDoesntSupportReading {
            get {
                return ResourceManager.GetString("StreamDoesntSupportReading", resourceCulture);
            }
        }
    }
}