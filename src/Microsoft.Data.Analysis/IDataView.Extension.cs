// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Microsoft.Data.Analysis
{
    public static class IDataViewExtensions
    {
        public static DataFrame ToDataFrame(this IDataView dataView)
        {
            DataViewSchema schema = dataView.Schema;
            List<DataFrameColumn> columns = new List<DataFrameColumn>(schema.Count);

            List<DataViewSchema.Column> activeColumns = new List<DataViewSchema.Column>();
            foreach (DataViewSchema.Column column in schema)
            {
                if (column.IsHidden)
                {
                    continue;
                }
                activeColumns.Add(column);
                DataViewType type = column.Type;
                if (type == BooleanDataViewType.Instance)
                {
                    columns.Add(new PrimitiveDataFrameColumn<bool>(column.Name, dataView.GetRowCount() ?? 0));
                }
                else if (type == NumberDataViewType.Byte)
                {
                    columns.Add(new PrimitiveDataFrameColumn<byte>(column.Name, dataView.GetRowCount() ?? 0));
                }
                else if (type == NumberDataViewType.Double)
                {
                    columns.Add(new PrimitiveDataFrameColumn<double>(column.Name, dataView.GetRowCount() ?? 0));
                }
                else if (type == NumberDataViewType.Single)
                {
                    columns.Add(new PrimitiveDataFrameColumn<float>(column.Name, dataView.GetRowCount() ?? 0));
                }
                else if (type == NumberDataViewType.Int32)
                {
                    columns.Add(new PrimitiveDataFrameColumn<int>(column.Name, dataView.GetRowCount() ?? 0));
                }
                else if (type == NumberDataViewType.Int64)
                {
                    columns.Add(new PrimitiveDataFrameColumn<long>(column.Name, dataView.GetRowCount() ?? 0));
                }
                else if (type == NumberDataViewType.SByte)
                {
                    columns.Add(new PrimitiveDataFrameColumn<sbyte>(column.Name, dataView.GetRowCount() ?? 0));
                }
                else if (type == NumberDataViewType.Int16)
                {
                    columns.Add(new PrimitiveDataFrameColumn<short>(column.Name, dataView.GetRowCount() ?? 0));
                }
                else if (type == NumberDataViewType.UInt32)
                {
                    columns.Add(new PrimitiveDataFrameColumn<uint>(column.Name, dataView.GetRowCount() ?? 0));
                }
                else if (type == NumberDataViewType.UInt64)
                {
                    columns.Add(new PrimitiveDataFrameColumn<ulong>(column.Name, dataView.GetRowCount() ?? 0));
                }
                else if (type == NumberDataViewType.UInt16)
                {
                    columns.Add(new PrimitiveDataFrameColumn<ushort>(column.Name, dataView.GetRowCount() ?? 0));
                }
                else if (type == TextDataViewType.Instance)
                {
                    columns.Add(new StringDataFrameColumn(column.Name, dataView.GetRowCount() ?? 0));
                }
                else
                {
                    throw new NotSupportedException(nameof(type));
                }
            }

            DataFrame ret = new DataFrame(columns);
            DataViewRowCursor cursor = dataView.GetRowCursor(activeColumns);
            while (cursor.MoveNext())
            {
                foreach (var column in activeColumns)
                {
                    columns[column.Index].AddValueUsingCursor(cursor, column);
                }
            }

            return ret;
        }
    }

}
